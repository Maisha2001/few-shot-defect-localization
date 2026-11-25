"""Local inspection and annotation API."""
from contextlib import closing
from hashlib import sha256
from hmac import compare_digest
import io
import json
import os
from pathlib import Path
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, StrictInt

from .data import file_hash
from .export import ARMS, session
from .model import MODEL_REVISION, pixels
from .reviews import Conflict, Store

class Engine:
    def __init__(self,directory):
        directory=Path(directory)
        m=json.loads((directory/"manifest.json").read_text())
        graph=directory/"inspection.onnx"
        if m["schema"]!=1 or m["encoder_revision"]!=MODEL_REVISION or tuple(m["arms"])!=ARMS or file_hash(graph)!=m["graph_sha256"]:
            raise ValueError("Inconsistent model manifest")
        self.session=session(graph)
        self.arms=list(ARMS)
        self.versions={arm:m["graph_sha256"]+":"+m["heads"][arm]["sha256"]+":"+arm for arm in self.arms}
        self.thresholds={arm:float(m["heads"][arm]["threshold"]) for arm in self.arms}
        if not all(0<=t<=1 for t in self.thresholds.values()):raise ValueError("Invalid thresholds")

    def infer(self,im,arm):
        scores,maps=self.session.run(["score","probability"],{"image":pixels(im)[None].numpy()})
        j=self.arms.index(arm)
        return float(scores[j]),maps[j]

class Annotation(BaseModel):
    model_config=ConfigDict(extra="forbid")
    request_id: UUID
    expected_revision: StrictInt=Field(ge=0)
    label: str
    runs: list[tuple[StrictInt,StrictInt]]=Field(max_length=1568)

    def check_mask(self):
        if self.label not in {"normal","defect"} or bool(self.runs)!=(self.label=="defect"):
            raise ValueError("Supply foreground runs for a defect and an empty mask for normal")
        previous=-1
        for start,length in self.runs:
            if start<0 or length<=0 or start<=previous or start+length>3136:
                raise ValueError("Use sorted, disjoint, merged runs within the 56 by 56 grid")
            previous=start+length

async def bounded_body(req,limit):
    buf=bytearray()
    async for part in req.stream():
        if len(buf)+len(part)>limit:raise HTTPException(413,"Input exceeds size limit")
        buf.extend(part)
    return bytes(buf)

def decode(body,kind):
    try:
        with Image.open(io.BytesIO(body)) as im:
            if im.format!={"image/png":"PNG","image/jpeg":"JPEG"}[kind] or getattr(im,"n_frames",1)!=1:
                raise ValueError("Expected one PNG or JPEG matching the content type")
            if im.width*im.height>6_000_000 or max(im.size)>6000:raise ValueError("Image exceeds pixel limit")
            if im.getexif().get(274,1)!=1:raise ValueError("Apply image orientation before upload")
            return im.convert("RGB")
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError) as err:
        raise HTTPException(422,"Unsupported or unreadable image") from err

def inspect(store,engine,body,kind,arm):
    digest=sha256(body).hexdigest();version=engine.versions[arm]
    key=sha256((digest+":"+version+":"+kind).encode()).hexdigest()
    row=store.get(key)
    if row:return response(row)
    im=decode(body,kind)
    score,heat=engine.infer(im,arm)
    if heat.shape!=(56,56) or not np.isfinite(heat).all() or not np.isfinite(score) or heat.min()<0 or heat.max()>1:
        raise RuntimeError("Invalid inference result")
    threshold=engine.thresholds[arm]
    # Rank unreviewed cases by proximity of the closest eight cells to the decision boundary.
    margins=np.abs(heat.ravel()-threshold)
    uncertainty=float(1-np.partition(margins,7)[:8].mean())
    image=io.BytesIO()
    clean=Image.frombytes("RGB",im.size,im.tobytes())
    clean.save(image,format="PNG")
    row=store.add((key,digest,version,*im.size,image.getvalue(),score,threshold,uncertainty,heat.astype("<f4").tobytes()))
    return response(row)

def response(row):
    row["probability"]=np.frombuffer(row.pop("heat"),dtype="<f4").reshape(56,56).tolist()
    return row

def create_app(engine=None,store=None,token=None):
    token=token if token is not None else os.environ.get("INSPECTION_TOKEN","")
    if len(token)<24 or not token.isascii():raise ValueError("Set a local ASCII bearer token of at least 24 characters")
    if engine is None:engine=Engine(os.environ["INSPECTION_MODEL"])
    if store is None:store=Store(os.environ["INSPECTION_DB"])
    app=FastAPI(title="Industrial inspection",docs_url=None,redoc_url=None,openapi_url=None)

    @app.middleware("http")
    async def authenticate(request,call_next):
        if not compare_digest(request.headers.get("authorization","").encode(),("Bearer "+token).encode()):
            return Response(status_code=401,headers={"WWW-Authenticate":"Bearer"})
        return await call_next(request)

    @app.get("/models")
    def models():return {"versions":engine.versions,"thresholds":engine.thresholds,"grid":[56,56]}

    @app.post("/cases")
    async def upload(request:Request,arm:str):
        if arm not in engine.arms:raise HTTPException(422,"Unknown model arm")
        kind=request.headers.get("content-type","").split(";")[0].lower()
        if kind not in {"image/png","image/jpeg"}:raise HTTPException(415,"Submit image/jpeg or image/png")
        return await run_in_threadpool(inspect,store,engine,await bounded_body(request,6<<20),kind,arm)

    @app.get("/cases/{case_id}")
    def case(case_id:str):
        row=store.get(case_id)
        if not row:raise HTTPException(404,"Unknown case")
        return response(row)

    @app.get("/cases/{case_id}/image")
    def image(case_id:str):
        row=store.get(case_id,image=True)
        if not row:raise HTTPException(404,"Unknown case")
        return Response(row["image"],media_type="image/png")

    @app.get("/queue")
    def queue(limit:int=Query(50,ge=1,le=200)):return store.queue(limit)

    @app.put("/cases/{case_id}/review")
    async def review(case_id:str,request:Request):
        try:
            a=Annotation.model_validate_json(await bounded_body(request,65536))
            a.check_mask()
            return await run_in_threadpool(store.review,case_id,str(a.request_id),a.expected_revision,a.label,a.runs)
        except (ValidationError,ValueError) as e:raise HTTPException(422,"Invalid annotation or mask geometry") from e
        except KeyError as e:raise HTTPException(404,"Unknown case") from e
        except Conflict as e:raise HTTPException(409,str(e)) from e

    @app.get("/reviews")
    def changes(after:int=Query(0,ge=0),limit:int=Query(100,ge=1,le=200)):
        return {"grid":[56,56],"encoding":"row-major start,length","reviews":store.changes(after,limit)}
    return app
