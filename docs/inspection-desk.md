# Inspection desk

Compare two models on the same image, inspect their 56 × 56 overlays, and paint or erase a review mask. The active and random models each used 32 source reviews and eight calibration images. Orange marks cells that reached each model’s own frozen threshold. Teal marks your annotation on the selected model.

## Run locally

From a copy of this repository, use Python 3.13 in a fresh environment. The following commands download the pinned DINOv2 encoder, combine it with the two included heads and start the demo. The encoder download is about 90 MB. The demo needs no benchmark archive or account.

```sh
python -m pip install -r requirements-service.txt
python -m inspection.download --model-only runs/encoder
python -m inspection.export --inputs runs/encoder --models models --report results/adaptation.json --output runs/desk-model
python -m inspection.demo --model runs/desk-model --database runs/reviews/desk.sqlite
```

Open **http://127.0.0.1:8026/** on the same machine. Enter the access key printed in the terminal. Choose **Capsules** and **Inspect sample**. The two panels highlight different areas using their calibrated thresholds. Select **Review model**, paint on that image and choose **Save review**. An empty mask records a normal review. Switching models restores the latest saved mask for that model. **Export reviews** downloads a JSON history for the local database. To revisit a reviewed image, select its sample again or upload the same original bytes.

Run the export once per output directory. To reopen an existing installation, use the last command with the same model and database paths. Stop the local server with Ctrl+C. Keep the database private if it contains your own images. Each annotation is versioned against the exact model that produced its case. Saving a review leaves the trained models fixed.

## Examples and interpretation

The three sample JPEGs are unmodified images from the VisA development partition. The capsules example was chosen to show both models producing visible foreground. The other examples show different products and include a normal image. These illustrations are separate from the final evaluation. See the [example manifest](../inspection/ui/examples/index.json) for the original paths and file hashes.

Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang and Onkar Dabeer released VisA under [CC BY 4.0](../inspection/ui/examples/LICENSE-DATASET.txt). See the [dataset source](https://github.com/amazon-science/spot-diff). The frozen DINOv2 encoder uses its [pinned Apache-2.0 weights](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056). The two included linear heads were trained on VisA. The image ranking score is available in each panel’s flagged-cell tooltip. The review queue prioritizes predictions nearest their model’s pixel threshold.

The release demo was exercised in a desktop browser and at 390 pixels wide, including drawing a mask, reopening model-specific revisions, clearing two queue entries and exporting both reviews. See the [API and latency measurements](service.md) for the underlying service.
