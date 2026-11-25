import importlib.util
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

if not all(importlib.util.find_spec(m) for m in ("onnx","onnxruntime","fastapi","httpx")):
    raise unittest.SkipTest("Install requirements-service.txt for API and export tests")

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import torch
from transformers import Dinov2Config, Dinov2Model

from inspection.adapt import PatchHead
from inspection.export import ARMS, Segmenter, session
from inspection.reviews import Conflict, Store
from inspection.serve import create_app

class FixtureEngine:
    arms=list(ARMS)
    versions={a:"fixture:"+a for a in ARMS}
    thresholds=dict.fromkeys(ARMS,.25)
    calls=0
    def infer(self,im,arm):
        self.calls+=1
        return .75,np.full((56,56),.5 if arm==ARMS[0] else .25,dtype=np.float32)

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db=Path(self.tmp.name)/"reviews.db"
        self.store=Store(self.db);self.engine=FixtureEngine()
        token="fixture-token-for-local-tests"
        self.app=create_app(self.engine,self.store,token)
        self.client=TestClient(self.app,headers={"Authorization":"Bearer "+token})
        self.addCleanup(self.client.close)
        buf=io.BytesIO()
        Image.new("RGB",(31,11),(255,100,0)).save(buf,format="PNG")
        self.body=buf.getvalue()

    def upload(self,arm=ARMS[0],body=None,kind="image/png"):
        return self.client.post("/cases",params={"arm":arm},content=self.body if body is None else body,
                                headers={"Content-Type":kind})

    def test_case_identity_includes_model_and_duplicate_reuses_output(self):
        one=self.upload().raise_for_status().json()
        two=self.upload().raise_for_status().json()
        other=self.upload(ARMS[1]).raise_for_status().json()
        self.assertEqual(one,two);self.assertNotEqual(one["id"],other["id"])
        self.assertEqual(self.engine.calls,2)
        self.assertEqual(len(one["probability"]),56)
        self.assertEqual(self.client.get("/queue").json()[0]["id"],other["id"])
        data=self.client.get("/cases/"+one["id"]+"/image").content
        self.assertEqual(Image.open(io.BytesIO(data)).size,(31,11))

    def test_access_and_input_validation(self):
        self.assertEqual(self.client.get("/models",headers={"Authorization":""}).status_code,401)
        self.assertEqual(self.upload(kind="text/plain").status_code,415)
        self.assertEqual(self.upload(kind="image/jpeg").status_code,422)
        self.assertEqual(self.upload(body=b"invalid").status_code,422)
        self.assertEqual(self.upload(body=b"x"*((6<<20)+1)).status_code,413)
        self.assertEqual(self.upload("missing").status_code,422)
        self.assertEqual(self.client.get("/cases/missing").status_code,404)

    def test_review_conflict_retry_and_incremental_export(self):
        key=self.upload().json()["id"];path=f"/cases/{key}/review"
        a={"request_id":str(uuid4()),"expected_revision":0,"label":"defect","runs":[[70,2]]}
        receipt=self.client.put(path,json=a).raise_for_status().json()
        self.assertEqual(self.client.put(path,json=a).json(),receipt)
        self.assertEqual(self.client.put(path,json=dict(a,runs=[[71,2]])).status_code,409)
        self.assertEqual(self.client.put(path,json=dict(a,request_id=str(uuid4()))).status_code,409)
        b=dict(a,request_id=str(uuid4()),expected_revision=1,label="normal",runs=[])
        self.assertEqual(self.client.put(path,json=b).json()["revision"],2)
        delta=self.client.get("/reviews",params={"after":receipt["seq"]}).json()["reviews"]
        self.assertEqual([r["revision"] for r in delta],[2])
        self.assertEqual(self.store.get(key)["revision"],2)
        self.assertEqual(Store(self.db).queue(10),[])
        self.assertEqual(self.store.changes(0,10)[0]["runs"],[[70,2]])

    def test_malformed_masks_are_rejected_without_persisting(self):
        key=self.upload().json()["id"]
        for values in ([], [[0,0]], [[3135,2]], [[2,2],[1,1]], [[4,1],[5,1]], [[True,1]]):
            data={"request_id":str(uuid4()),"expected_revision":0,"label":"defect","runs":values}
            self.assertEqual(self.client.put(f"/cases/{key}/review",json=data).status_code,422)
        self.assertEqual(self.store.changes(0,5),[])

    def test_concurrent_reviews_cannot_overwrite_each_other(self):
        key=self.upload().json()["id"]
        def send(_):
            try:return self.store.review(key,str(uuid4()),0,"normal",[])["revision"]
            except Conflict:return 0
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(sorted(executor.map(send,range(4))),[0,0,0,1])
        self.assertEqual(len(self.store.changes(0,20)),1)

    def test_exported_two_head_graph_matches_eager_on_distinct_images(self):
        torch.manual_seed(47)
        torch.set_num_threads(2)
        cfg=Dinov2Config(image_size=224,patch_size=14,hidden_size=24,num_hidden_layers=1,num_attention_heads=3)
        heads=[PatchHead([.1]*23+[.8],.2),PatchHead([-.2]*23+[.4],-.5)]
        model=Segmenter(Dinov2Model(cfg),heads).eval()
        graph=Path(self.tmp.name)/"tiny.onnx"
        image=torch.rand(1,3,224,224)
        with torch.inference_mode():
            torch.onnx.export(model,(image,),str(graph),opset_version=17,dynamo=False,
                              input_names=["image"],output_names=["score","probability"])
            runtime=session(graph)
            for sample in (image,1-image):
                a=model(sample);b=runtime.run(None,{"image":sample.numpy()})
                for u,v in zip(a,b):np.testing.assert_allclose(u.numpy(),v,rtol=1e-5,atol=1e-5)
