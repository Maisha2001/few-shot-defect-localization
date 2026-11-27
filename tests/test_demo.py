import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

if not all(importlib.util.find_spec(m) for m in ("fastapi","httpx","onnx","onnxruntime")):
    raise unittest.SkipTest("Install service dependencies for the browser workspace")

from fastapi.testclient import TestClient
from inspection.data import file_hash
from inspection.demo import workspace
from inspection.reviews import Store
from test_service import FixtureEngine


class DemoTests(unittest.TestCase):
    def test_uploaded_pair_annotation_and_private_routes(self):
        with tempfile.TemporaryDirectory() as temp:
            key = "local-workspace-fixture-key"
            with TestClient(workspace(FixtureEngine(),Store(Path(temp)/"db"),key)) as client:
                self.assertEqual(client.get("/").status_code,200)
                self.assertEqual(client.get("/api/models").status_code,401)
                self.assertEqual(client.get("/api/cases/x/annotation").status_code,401)
                self.assertEqual(client.get("/",headers={"host":"other.example"}).status_code,400)
                examples = client.get("/examples/index.json").json()
                base = Path(__file__).resolve().parents[1]
                for sample in examples:
                    self.assertIn(sample["source"].split("/")[0], {"capsules","macaroni1","macaroni2"})
                    self.assertEqual(file_hash(base/"inspection/ui/examples"/sample["file"]),sample["sha256"])
                photo = client.get("/examples/"+examples[0]["file"]).content
                client.headers["Authorization"] = "Bearer "+key
                ids = {}
                for arm in FixtureEngine.arms:
                    row = client.post("/api/cases",params={"arm":arm},content=photo,
                                      headers={"Content-Type":"image/jpeg"}).raise_for_status().json()
                    ids[arm] = row["id"]
                url = "/api/cases/"+ids["active-32"]
                self.assertEqual(client.get(url+"/siblings").json(),ids)
                self.assertEqual(client.get(url+"/annotation").json()["revision"],0)
                payload = {"expected_revision":0,"request_id":str(uuid4()),"label":"defect","runs":[[0,3]]}
                client.put(url+"/review",json=payload).raise_for_status()
                self.assertEqual(client.get(url+"/annotation").json(),{"revision":1,"label":"defect","runs":[[0,3]]})
                self.assertEqual(len(client.get("/api/queue").json()),1)
                self.assertEqual(client.get("/api/cases/missing/siblings").status_code,404)

    def test_published_head_digests_and_thresholds(self):
        base = Path(__file__).resolve().parents[1]
        spec = json.loads((base/"protocols/cross-product.json").read_text())
        for arm, metadata in spec["heads"].items():
            path = base/"models"/(arm+".json")
            self.assertEqual(file_hash(path),metadata["sha256"])
            self.assertEqual(json.loads(path.read_text())["threshold"],metadata["threshold"])
