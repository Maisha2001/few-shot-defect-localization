# Cross-product evaluation

We froze the evaluation plan, both model heads, pixel thresholds, encoder, graph and reference bank before opening the reserved product groups. The active head was already designated for deployment in the development experiment. The random head uses the same review budget. Both use the same eight cashew images for calibration. The reference bank uses normal examples only.

All 450 official test images from chewing gum, fryum and pipe fryum were included. Each product contributes 100 anomalous and 50 normal images. The run compared every test image against 1,000 earlier reference, acquisition, calibration and development images using the prespecified thumbnail check. It found zero near matches under that check. Predictions for all models were saved before reading final masks. Native-pixel weights preserve mixed foreground and background within the 56 × 56 cells.

| Product | Defect pixels | Active pixel AP | Random pixel AP | Reference pixel AP | Active pixel recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chewing gum | 1,271,411 | 64.72% | 58.24% | 41.12% | 74.97% |
| Fryum | 5,186,101 | 14.22% | 13.72% | 11.67% | 23.06% |
| Pipe fryum | 2,810,275 | 27.43% | 34.05% | 47.35% | 0.93% |
| Pooled | 9,267,787 | 20.85% | 19.72% | 12.76% | 23.47% |

The primary endpoint was pooled native-pixel average precision (AP). Defects cover 1.37% of the 675,083,400 native pixels. Pooled image AUROC was 0.706 for active, 0.636 for random and 0.525 for the reference bank. The frozen active threshold produced 21.70% precision and 0.226 Dice. It marked foreground on 50 of 150 normal images, including 48 of the 50 normal fryum images. It found zero overlap on 120 of 300 anomalous images. On pipe fryum its image AUROC was 0.956, yet its mask missed all defect pixels on 81 of 100 anomalous images. Ranking an image and locating the defect are distinct decisions.

The active and random heads had pooled pixel Brier scores of 0.01305 and 0.01284. A single acquisition order, calibration on one product, coarse masks and three test groups limit generalization. The simple reference bank led pixel AP on pipe fryum. These results do not establish a reliable production inspection threshold or a general advantage for active selection.

## Timing and verification

The deployed two-head graph took 43.5 ms median and 49.4 ms p95 on four CPU threads for 450 preprocessed, single-image calls, including the first call. This excludes image decoding, overlay rendering, storage and transport. The earlier [local service benchmark](service.md) measured 229.9 ms p95 for new serial HTTP inspections. The final prediction stage took 76.1 seconds wall time, including interpreter startup and generating reference-bank scores. Its internal prediction timer measured 71.4 seconds. Peak worker memory was not established for this final run.

An independent audit compared all 300 original masks and the normal-image grids with the saved native-pixel counts. It checked 182 values across 12 model and product summaries and latency statistics. The maximum absolute discrepancy was 2.23 × 10⁻¹⁶. [Frozen protocol](../results/final-protocol.json) · [Full measurements](../results/final.json) · [Audit](../results/final-audit.json)

## Reproduce in a fresh directory

Follow the [baseline instructions](../README.md#reproduce) to obtain the original VisA archive and reference bank. Install the service dependencies and export the two frozen heads as described in the [demo setup](inspection-desk.md). Use fresh output paths. The frozen evaluation expects the exact pinned graph and reference-bank hashes. Use the recorded versions, CPU and float32 execution. A different graph digest stops the evaluation.

```sh
python -m inspection.final freeze --inputs data/visa --exported runs/desk-model --bank runs/baseline/reference-bank.npy --output runs/cross-product
python -m inspection.final predict --inputs data/visa --exported runs/desk-model --bank runs/baseline/reference-bank.npy --output runs/cross-product
python -m inspection.final score --inputs data/visa --output runs/cross-product
```

The first step locks the inputs and evaluator. The second saves image scores, maps and sample order. The third verifies their digests before reading masks. Release records include timestamps and measured runtime. These vary between executions.

To record the published aggregates locally in Weights & Biases, install `requirements-tracking.txt` in a separate environment and run `python -m inspection.track --output runs/tracking`. The tracker runs in offline mode.
