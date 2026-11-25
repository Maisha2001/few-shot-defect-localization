# Local image inspection API

Upload a JPEG or PNG, get a 56 × 56 probability map, and review the cases nearest the selected model's pixel threshold. SQLite keeps the input image, model version, map and annotation history on your machine. Reviewed labels can be exported for a later training run.

## Setup

Use Python 3.13 in the project environment. First reproduce the [few-shot experiment](adaptation.md) to create the two head files and their report. The exporter verifies the source weights and trained head hashes. It writes one two-head ONNX graph and a manifest to a fresh directory.

```sh
python -m pip install -r requirements-service.txt
python -m inspection.export --inputs data/visa --models runs/adaptation --report runs/adaptation/report.json --output runs/exported
```

In PowerShell, start the service on loopback using the paths below. Keep the token private. The review database contains uploaded images and annotations.

```powershell
$env:INSPECTION_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:INSPECTION_MODEL = "runs/exported"
$env:INSPECTION_DB = "runs/reviews/inspection.sqlite"
python -m uvicorn inspection.serve:create_app --factory --host 127.0.0.1 --port 8026
```

Use a client with the same token. This example submits an image, marks an example foreground run, and fetches the revisions. Supply the annotation that matches the actual image.

```python
import os, uuid
from pathlib import Path
import httpx

with httpx.Client(base_url="http://127.0.0.1:8026", trust_env=False,
                  headers={"Authorization": "Bearer " + os.environ["INSPECTION_TOKEN"]}) as client:
    case = client.post("/cases", params={"arm": "random-32"},
                       content=Path("inspection.jpg").read_bytes(),
                       headers={"Content-Type": "image/jpeg"}).raise_for_status().json()
    payload = {"request_id": str(uuid.uuid4()), "expected_revision": case["revision"],
               "label": "defect", "runs": [[100, 3]]}
    receipt = client.put(f"/cases/{case['id']}/review", json=payload).raise_for_status().json()
    print(receipt, client.get("/reviews", params={"after": 0}).raise_for_status().json())
```

## Responses and review flow

- `GET /models` lists model versions and thresholds. `POST /cases?arm=random-32` and `arm=active-32` accept one matching PNG or JPEG, up to 6 MiB and six million pixels. Apply EXIF orientation before upload. Replaying the same bytes with the same model returns the stored case.
- `GET /cases/{id}` returns the probability grid, dimensions, version, threshold, image ranking score, review priority and current annotation revision. `GET /cases/{id}/image` returns the RGB image as PNG. The model receives the same full-frame warp as the baseline. Multiply a source pixel's normalized coordinates by 56 and take the floor to identify its cell.
- `GET /queue?limit=50` lists pending cases ordered by closeness to the model decision threshold across the eight closest cells. This is a review heuristic. The image ranking score does not estimate the probability that the entire image is defective.
- `PUT /cases/{id}/review` accepts a complete replacement annotation and the expected revision. Masks use zero-based row-major runs of `[start,length]` on the 56 × 56 grid. Normal annotations have an empty list. Defect masks require sorted, merged, in-range runs. An identical request ID and payload replays its receipt. A stale revision or reused ID with different content returns HTTP 409. Concurrent writers cannot overwrite an accepted revision.
- `GET /reviews?after=0&limit=100` returns ordered changes. For the next page use the last sequence number as `after`. Each revision includes its source digest, model version, dimensions and mask. An annotation updates the review store only. The model graph remains frozen.

## Measured behavior

The fixed-shape ONNX graph embeds the pretrained encoder, patch normalization, both heads, bilinear upsampling and probability calibration. The server uses ONNX Runtime's native CPU provider with four intra-op threads. The export fixes batch size 1 and input dimensions 3 × 224 × 224. Verify the graph hash when copying the local bundle.

The comparison used 48 development images chosen by a fixed hash order. It tested 96 image scores and 301,056 heatmap cells against eager PyTorch and saved development predictions. The largest map difference from the saved predictions was 0.00000942 at a prespecified absolute tolerance of 0.00002. Zero cell decisions changed at the fitted thresholds.

| Local measurement | Requests | Median | p95 |
| --- | ---: | ---: | ---: |
| Eager PyTorch, two heads | 48 | 65.3 ms | 70.7 ms |
| ONNX Runtime, two heads | 48 | 48.2 ms | 51.0 ms |
| New HTTP inspection, serial | 24 | 210.0 ms | 229.9 ms |
| Repeated HTTP inspection, serial | 24 | 7.3 ms | 8.9 ms |
| New HTTP inspection, four clients | 24 | 260.8 ms | 299.0 ms |

Inference timings alternate backend order on preprocessed tensors after warmup. The HTTP test includes decoding, PNG encoding, SQLite writes or lookups, transfer and JSON parsing. Under four clients, new requests completed at 14.8 per second on this machine. These short loopback runs characterize this configuration. A handwritten inference kernel was unnecessary for the measured gain. SQLite retention and local token distribution are the operator's responsibility.

The benchmark restarts the service, reopens the same database and checks a retried annotation. The annotation is authored test input. To reproduce with a new output directory:

```sh
python -m inspection.benchmark --inputs data/visa --models runs/adaptation --report runs/adaptation/report.json --exported runs/exported --out runs/service-check
```

Full results are in the [service report](../results/service.json). See the upstream [PyTorch ONNX exporter](https://docs.pytorch.org/docs/stable/onnx.html) and [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).
