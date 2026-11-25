"""Compare exported predictions to saved evidence, then measure loopback inspection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from hashlib import sha256
import io
import json
from pathlib import Path
import secrets
import socket
import threading
import time
from uuid import uuid4

import httpx
import numpy as np
from PIL import Image
import torch
import uvicorn

from .data import Archive, file_hash, protocol
from .evaluate import ARCHIVE_SHA, write_json
from .export import ARMS, Segmenter, load_heads
from .model import load_encoder,pixels
from .reviews import Store
from .serve import Engine,create_app

def timing(values):
    return {"n":len(values),"p50_ms":float(1000*np.median(values)),"p95_ms":float(1000*np.quantile(values,.95))}

@contextmanager
def running(app):
    sock=socket.socket()
    sock.bind(("127.0.0.1",0))
    server=uvicorn.Server(uvicorn.Config(app,log_level="error",access_log=False))
    thread=threading.Thread(target=server.run,kwargs={"sockets":[sock]},daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:break
            if not thread.is_alive():raise RuntimeError("Server exited at startup")
            time.sleep(.05)
        if not server.started:raise TimeoutError("Server startup timed out")
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit=True
        thread.join(15)
        sock.close()
        if thread.is_alive():raise RuntimeError("Service did not stop")

def run(inputs,models,report_path,exported,out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    report=json.loads(report_path.read_text())
    if file_hash(inputs/"VisA.tar")!=ARCHIVE_SHA:raise ValueError("Unexpected archive")
    _,dev=protocol(inputs/"1cls.csv")
    ranked=sorted(range(len(dev)),key=lambda i:sha256(("service-v1/"+dev[i].image).encode()).digest())
    selected=ranked[:48]
    with Archive(inputs/"VisA.tar",[dev[i] for i in selected]) as archive:
        raw=[archive.tar.extractfile(archive.members[dev[i].image]).read() for i in selected]
    images=[Image.open(io.BytesIO(b)).convert("RGB") for b in raw]
    eager=Segmenter(load_encoder(inputs),load_heads(models,report)).eval()
    engine=Engine(exported)
    if json.loads((exported/"manifest.json").read_text())["report_sha256"]!=file_hash(report_path):
        raise ValueError("The exported graph is bound to a different report")
    keys=("score","probability")
    diff={k:0. for k in keys};saved={k:0. for k in keys};flips=0
    eager_times=[];ort_times=[];native={}
    with torch.inference_mode(),ExitStack() as stack:
        traces={}
        for arm in ARMS:
            path=models/(arm+".npz")
            if file_hash(path)!=report["arms"][arm]["predictions_sha256"]:raise ValueError("Development output changed")
            f=stack.enter_context(np.load(path))
            traces[arm]=(f["scores"][selected],f["prob"][selected])
        tensor=pixels(images[0])[None]
        eager(tensor);engine.session.run(None,{"image":tensor.numpy()})
        for j,im in enumerate(images):
            tensor=pixels(im)[None];arg={"image":tensor.numpy()}
            def torch_run():
                t=time.perf_counter();v=eager(tensor);eager_times.append(time.perf_counter()-t)
                return tuple(s.numpy() for s in v)
            def ort_run():
                t=time.perf_counter();v=engine.session.run(None,arg);ort_times.append(time.perf_counter()-t)
                return v
            if j%2:actual=ort_run();expected=torch_run()
            else:expected=torch_run();actual=ort_run()
            native[j]=actual
            for k,u,v in zip(keys,actual,expected):
                diff[k]=max(diff[k],float(np.max(np.abs(u-v))))
            for a,arm in enumerate(ARMS):
                sc,pr=traces[arm]
                saved["score"]=max(saved["score"],float(abs(sc[j]-actual[0][a])))
                saved["probability"]=max(saved["probability"],float(np.max(np.abs(pr[j]-actual[1][a]))))
                threshold=engine.thresholds[arm]
                flips+=int(np.count_nonzero((pr[j]>=threshold)!=(actual[1][a]>=threshold)))
    if max(*diff.values(),*saved.values())>2e-5:raise ValueError(f"Export exceeded absolute tolerance: {diff}, {saved}")
    token=secrets.token_urlsafe(32);db=out/"inspection.sqlite"
    headers={"Authorization":"Bearer "+token};stored=[]
    http_times={};largest_http_diff=0.
    with running(create_app(engine,Store(db),token)) as url,httpx.Client(base_url=url,headers=headers,timeout=60,trust_env=False) as client:
        def send(j):
            t=time.perf_counter()
            r=client.post("/cases",params={"arm":ARMS[j%2]},content=raw[j],headers={"Content-Type":"image/jpeg"})
            r.raise_for_status();body=r.json();elapsed=time.perf_counter()-t
            actual=native[j]
            delta=max(abs(body["score"]-actual[0][j%2]),np.max(np.abs(np.asarray(body["probability"])-actual[1][j%2])))
            return body["id"],elapsed,float(delta)
        for mode,indices in (("serial_new",range(24)),("cached",range(24)),("four_clients_new",range(24,48))):
            t=time.perf_counter()
            if mode=="four_clients_new":
                with ThreadPoolExecutor(max_workers=4) as exe: rows=list(exe.map(send,indices))
            else:rows=list(map(send,indices))
            elapsed=time.perf_counter()-t
            http_times[mode]=dict(**timing([r[1] for r in rows]),requests_per_second=len(rows)/elapsed)
            largest_http_diff=max(largest_http_diff,max(r[2] for r in rows))
            ids=[r[0] for r in rows]
            if mode=="cached":
                if ids!=stored:raise ValueError("Replay created new cases")
            else:stored.extend(ids)
        queue=client.get("/queue").raise_for_status().json()
        if len(queue)!=48 or len(set(stored))!=48:raise ValueError("Inspection queue count differs")
        # An authored annotation exercises storage. It supplies no supervision to the experiment.
        item=queue[0]["id"];body={"request_id":str(uuid4()),"expected_revision":0,"label":"defect","runs":[[100,3]]}
        receipt=client.put(f"/cases/{item}/review",json=body).raise_for_status().json()
        if client.put(f"/cases/{item}/review",json=body).raise_for_status().json()!=receipt:raise ValueError("Replay differed")
    with running(create_app(Engine(exported),Store(db),token)) as url,httpx.Client(base_url=url,headers=headers,trust_env=False) as client:
        changes=client.get("/reviews").raise_for_status().json()["reviews"]
        if len(changes)!=1 or changes[0]["case_id"]!=item or changes[0]["runs"]!=body["runs"] or len(client.get("/queue").json())!=47:
            raise ValueError("Restart changed reviewed state")
    if largest_http_diff>2e-5:raise ValueError("HTTP inference mismatch")
    result={"protocol":"visa-inspection-service-v1","summary":{"development_images":48,"heads":2,
            "checked_heatmap_cells":48*2*3136,"max_absolute_vs_eager":diff,"max_absolute_vs_saved":saved,
            "threshold_cell_disagreements":flips,"max_http_absolute_difference":largest_http_diff,
            "eager_cpu":timing(eager_times),"onnx_cpu":timing(ort_times),"http":http_times},
            "explicit_absolute_tolerance":2e-5,"onnx_sha256":file_hash(exported/"inspection.onnx"),
            "export_manifest":json.loads((exported/"manifest.json").read_text()),
            "test_receipts":{"unique_cases":48,"annotation_revisions_after_restart":1,"pending_after_restart":47},
            "selected_indices":selected,"threads":4}
    write_json(out/"report.json",result)
    print(json.dumps(result["summary"],indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("inputs","models","report","exported","out"):p.add_argument("--"+k,type=Path,required=True)
    args=p.parse_args()
    run(args.inputs,args.models,args.report,args.exported,args.out)
