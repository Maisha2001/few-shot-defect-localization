# Few-Shot Industrial Defect Localization

[Run the inspection desk](docs/inspection-desk.md) to compare two model overlays on industrial images, paint defect masks and export a versioned review history. The local demo includes three VisA development images and accepts JPEG or PNG uploads.

On **450 images from three held-out products**, the active and random models reached **20.85% and 19.72% pixel average precision**, respectively. The normal-reference baseline reached 12.76%. At its frozen threshold the active model detected only 0.93% of pipe-fryum defect pixels. The two-head model took 49.4 ms p95 per preprocessed image on four CPU threads. [Cross-product evaluation, per-product results and limitations](docs/cross-product.md)

The [local inspection API](docs/service.md) returns a defect probability map for an uploaded image, queues uncertain cases and keeps a revision history of human annotations. Model versions cover both 32-review segmentation heads. On 48 development images, the ONNX export agreed with the original models on every thresholded grid cell.

A [few-shot segmentation experiment](docs/adaptation.md) compares active and random annotation selection across four budgets. With 32 source reviews and eight calibration reviews, random selection reached 19.33% pixel average precision on three unseen products. Active selection reached 3.98% at the same cost.

The first experiment provides a normal-reference baseline for inspecting products absent from the reference set. It builds a memory bank from frozen DINOv2 image patches and returns an anomaly score and a spatial score map for each development image.

## Development results

The reference bank uses 192 normal VisA images. The development set contains all 560 official test images from three different products, including 300 with annotated defects. Higher scores indicate greater distance from normal reference patches.

| Product | Images | Image AUROC | Pixel AUROC | Pixel average precision |
| --- | ---: | ---: | ---: | ---: |
| Capsules | 160 | 0.6053 | 0.8283 | 1.773% |
| Macaroni 1 | 200 | 0.4975 | 0.8396 | 0.146% |
| Macaroni 2 | 200 | 0.4570 | 0.9314 | 0.207% |
| Pooled | 560 | 0.5481 | 0.9113 | 1.502% |

Only 0.121% of the 840 million evaluated pixels belong to defects. The maps concentrate some defect signal, but their low average precision limits localization. On both macaroni products, image ranking was at or below chance. Background and product appearance can drive distance from the reference bank. The experiment supplies development evidence for a benchmark with strong product shift. It does not establish an operating threshold or factory inspection performance.

## Reproduce

Use Python 3.13 inside a fresh virtual environment. The downloader needs about 2.1 GB for the original archive and weights. It verifies the pinned checksums. The evaluator uses four CPU threads and writes to a new output directory.

```sh
python -m pip install -r requirements.txt
python -m inspection.download data/visa
python -m inspection.evaluate --archive data/visa/VisA.tar --split data/visa/1cls.csv --model data/visa --output runs/baseline
python -m inspection.verify runs/baseline/report.json --artifacts runs/baseline
python -m unittest discover -s tests -v
```

The sample manifest identifies the images in score order. The local `development.npz` contains image scores, 56 × 56 heatmaps, and native pixel counts for each cell. The evaluator reads selected archive members directly. Data, weights, and per-image predictions stay local. The published [report](results/baseline.json) includes all group metrics and input and output hashes.

## Protocol

The source is the authors’ one-class VisA split. All four PCB types share the reference partition, both macaroni types share the development partition, and both fryum types share the reserved partition.

| Partition | Products | Use |
| --- | --- | --- |
| Reference | Candle, cashew, PCB 1–4 | First 32 normal training images per product in SHA-256 path order |
| Development | Capsules, macaroni 1 and 2 | All official test images and their masks |
| Reserved | Chewing gum, fryum, pipe fryum | Held out for cross-product evaluation |

The split keeps every acquisition of a given product together. As a supplementary check, each development image is compared to every selected reference image using a 64-bit difference hash and a 32 × 32 grayscale thumbnail. A match within four bits and three gray levels RMS stops the run. This check found zero matches. Acquisition or camera identities beyond the product labels are unavailable in the split CSV.

Images are warped to 224 × 224 to retain the full field of view, then normalized with ImageNet channel statistics. DINOv2 produces 256 local 384-dimensional unit vectors. Hash ordering selects 16 spatial positions per reference image, for 3,072 reference patches. The score for a query patch is one minus its largest cosine similarity to the bank. The image score averages the four highest patch distances.

Bilinear interpolation maps the 16 × 16 patch distances onto a 56 × 56 grid. Each native pixel receives the score of its grid cell. Mixed cells contribute both positive and negative pixel counts to the metrics, preserving small defects and all normal-image pixels. Pooling combines examples across products and can mask weak individual results. The coarse spatial grid also limits boundary detail.

The recorded run took 45.5 seconds to score the development set on four CPU threads, including image and mask decoding and duplicate checks. Loading, fitting and verifying inputs took 18.7 seconds. Runtime varies by machine.

## Sources

VisA is distributed by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer under CC BY 4.0. This implementation selects products and normal reference images and converts nonzero mask values to binary defect labels. See the [VisA dataset and paper](https://github.com/amazon-science/spot-diff/tree/2a692ab575001cbde74d402d897a7286086c6199) and [dataset license](https://github.com/amazon-science/spot-diff/blob/2a692ab575001cbde74d402d897a7286086c6199/LICENSE-DATASET).

DINOv2 ViT-S/14 is used as a frozen encoder. The original weights are distributed under Apache 2.0. See the [pinned model card](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056).
