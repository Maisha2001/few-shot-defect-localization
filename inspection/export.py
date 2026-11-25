"""Export the frozen encoder and two segmentation heads as one fixed-size CPU graph."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F

from .adapt import PatchHead
from .data import file_hash
from .evaluate import write_json
from .model import MODEL_REVISION, load_encoder

ARMS = ("random-32","active-32")

class Segmenter(torch.nn.Module):
    def __init__(self, encoder, heads):
        super().__init__()
        self.encoder=encoder
        self.register_buffer("weight",torch.tensor([h.weight for h in heads],dtype=torch.float32))
        for name in ("bias","slope","offset"):
            self.register_buffer(name,torch.tensor([getattr(h,name) for h in heads],dtype=torch.float32))

    def forward(self, image):
        patches=F.normalize(self.encoder(pixel_values=image).last_hidden_state[:,1:],dim=-1)
        raw=(patches @ self.weight.T + self.bias).transpose(1,2)
        score=torch.sigmoid(raw.topk(4,dim=-1).values.mean(-1)*self.slope+self.offset)
        grid=F.interpolate(raw.reshape(1,2,16,16),size=(56,56),mode="bilinear",align_corners=False)
        prob=torch.sigmoid(grid*self.slope.view(1,2,1,1)+self.offset.view(1,2,1,1))
        return score[0],prob[0]

def load_heads(directory,report):
    heads=[]
    for arm in ARMS:
        path=directory/(arm+".json")
        if file_hash(path)!=report["arms"][arm]["head_sha256"]:
            raise ValueError(f"Head differs from the supplied experiment report: {arm}")
        heads.append(PatchHead(**json.loads(path.read_text())))
    return heads

def session(graph):
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    return ort.InferenceSession(str(graph),sess_options=options,providers=["CPUExecutionProvider"])

def export(inputs, models, report_path, output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    report=json.loads(report_path.read_text())
    if report["model_revision"]!=MODEL_REVISION:raise ValueError("Encoder revision differs")
    heads=load_heads(models,report)
    model=Segmenter(load_encoder(inputs),heads).eval()
    graph=output/"inspection.onnx"
    dummy=torch.zeros(1,3,224,224)
    with torch.inference_mode():
        torch.onnx.export(model,(dummy,),str(graph),input_names=["image"],output_names=["score","probability"],
                          opset_version=17,dynamo=False,external_data=False)
        expected=model(dummy)
    onnx.checker.check_model(str(graph))
    result=session(graph).run(None,{"image":dummy.numpy()})
    for a,b in zip(expected,result):
        np.testing.assert_allclose(a.numpy(),b,atol=2e-5,rtol=2e-5)
    manifest={"schema":1,"encoder_revision":MODEL_REVISION,"opset":17,"graph_sha256":file_hash(graph),
              "report_sha256":file_hash(report_path),"arms":list(ARMS),
              "heads":{a:{"sha256":report["arms"][a]["head_sha256"],"threshold":h.threshold} for a,h in zip(ARMS,heads)}}
    write_json(output/"manifest.json",manifest)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("inputs","models","report","output"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(export(a.inputs,a.models,a.report,a.output),indent=2))
