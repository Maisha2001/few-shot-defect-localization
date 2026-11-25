# Few-shot segmentation under product shift

A frozen DINOv2 encoder supplies a 384-dimensional feature for each 14 × 14 patch. A regularized linear foreground classifier learns from native-pixel foreground occupancy per patch. Class balancing gives defects sufficient training weight. If a round contains only normal images, distance from the mean background embedding supplies a fallback score.

The acquisition pool contains 48 benchmark images per product from candle and PCB 1–4. A fixed SHA-256 ordering of image paths chooses the pool and supplies the random control. Both arms begin with the same four images. The active arm then prioritizes uncertain patch margins and cosine distance from already selected image descriptors. It refits at 4, 8, 16 and 32 cumulative reviews. Each normal review counts toward the budget. This simulates full-image annotation at equal cost per image.

Eight separate cashew images (five anomalous) calibrate the monotone sigmoid and set a pixel decision threshold by maximum calibration Dice. These eight reviews are additional to every acquisition budget. Calibration reuses those same eight images at each round. The calibrated map predicts foreground per native pixel. The image score aggregates the four highest patch logits and is used to rank images. It is not an image-level defect probability.

| Source + calibration reviews | Random pixel AP | Active pixel AP | Random Dice | Active Dice |
| --- | ---: | ---: | ---: | ---: |
| 4 + 8 | 12.31% | 12.31% | 0.164 | 0.164 |
| 8 + 8 | 7.48% | 19.82% | 0.108 | 0.206 |
| 16 + 8 | 6.38% | 3.66% | 0.149 | 0.078 |
| 32 + 8 | 19.33% | 3.98% | 0.164 | 0.090 |

Each row uses the same 560 development images and 840 million native pixels from capsules and the two macaroni products. The frozen normal-reference baseline achieved 1.50% pixel AP. At 32 reviews, pixel Brier scores were 0.001079 for random and 0.001203 for active. For context, always predicting background gives 0.001205 on this set. Sparse foreground makes small calibration errors easy to miss in aggregate. The [full results](../results/adaptation.json) include per-product scores, 10-bin calibration error, thresholded Dice, selected indices, timing and artifact hashes. Pooled scores can differ substantially from within-product scores.

All eight heads and per-image outputs were saved before aggregating development metrics. The designated active-32 head was specified in the experiment code. The budget comparison uses one selection order and one calibration product. The budget counts images, not annotation minutes. Product shift, sparse flaws and coarse heatmaps limit transfer. Chewing gum, fryum and pipe fryum remain reserved.

## Reproduce

Follow the baseline setup in the [README](../README.md), including its local evidence files. Run the stages below with new output directories. They verify their inputs and use local weights. The cache stores normalized patch features for the source pool, calibration and development images. Development supervision comes from the verified baseline count arrays. During acquisition, only requested source masks and the calibration masks are decoded.

```sh
python -m inspection.experiment prepare --inputs data/visa --baseline runs/baseline --cache runs/features
python -m inspection.experiment train --inputs data/visa --baseline runs/baseline --cache runs/features --out runs/adaptation
```

Each head JSON contains the patch weights, bias, sigmoid parameters and threshold. Each prediction archive contains image scores and 56 × 56 probability maps in baseline development order. The cached input manifest and report bind the artifacts to SHA-256 digests.
