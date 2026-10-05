# MVTec AD baseline — 2026-10-01

## Run

The normal-only PatchCore baseline completed on all 15 categories: 3,629 normal training images and 1,725 test images. A toothbrush-only smoke run completed first. Both used the same settings; no device or batch-size fallback was needed.

| Setting | Value |
| --- | --- |
| Backbone | ImageNet-pretrained Wide ResNet-50-2 (`IMAGENET1K_V2`), frozen |
| Features | `layer2` and `layer3`, random projection to 256 dimensions |
| Memory | 2,048 uniformly sampled normal patches per category; seed 42 |
| Images / batch | 256 × 256 / 8 |
| Device | Apple M4 MPS, 24 GiB unified memory |
| Environment | Python 3.12.13; PyTorch 2.14.0; torchvision 0.29.0; Prefect 3.8.7 |
| Source | `main` `2b75255`; category-run `c9d7247` ([PR #7](https://github.com/keremoztopuz/industrial-anomaly-project/pull/7)); visualization `6b78fb9` ([PR #8](https://github.com/keremoztopuz/industrial-anomaly-project/pull/8)) |
| Runtime | Toothbrush smoke: 57.17 s, including first backbone download; all categories: 194.57 s |

The model stores normal features and uses nearest-neighbor distances for anomaly scoring. Its patch sampling is random, rather than PatchCore's greedy coreset selection. No model weights were fine-tuned. AUROC uses every test image and its resized ground-truth mask; scores range from 0 to 1, with higher better.

These are results from one baseline run; no hyperparameters were tuned and no uncertainty estimate was computed.

## Results

| Category | Image AUROC | Pixel AUROC |
| --- | ---: | ---: |
| bottle | 0.9802 | 0.9807 |
| cable | 0.7586 | 0.9360 |
| capsule | 0.7335 | 0.9717 |
| carpet | 0.9262 | 0.9786 |
| grid | 0.6850 | 0.9197 |
| hazelnut | 0.9600 | 0.9807 |
| leather | 0.9997 | 0.9917 |
| metal_nut | 0.8915 | 0.9391 |
| pill | 0.7717 | 0.9169 |
| screw | 0.5919 | 0.9687 |
| tile | 0.8669 | 0.9107 |
| toothbrush | 0.8889 | 0.9852 |
| transistor | 0.6804 | 0.7865 |
| wood | 0.9649 | 0.9249 |
| zipper | 0.8464 | 0.9667 |
| **Macro mean (equal category weight)** | **0.8364** | **0.9439** |

The toothbrush smoke run gave image AUROC 0.8889 and pixel AUROC 0.9852, matching the full run. All 15 bank files and 15 metric entries were produced; every AUROC was finite and within [0, 1].

## Visual review

Panels show the first two normal and first two defective test files in path order for each category. I inspected `bottle`, `cable`, `grid`, `leather`, `screw`, and `transistor`. The `leather` overlay concentrates on the two visible defects, and the larger broken `bottle` region is highlighted. In `screw` and `grid`, normal object edges and texture also receive strong responses; the defect often has little contrast against them. `transistor` responses spread across normal and defective parts, consistent with its low pixel AUROC. Panels normalize overlay intensity across their four selected examples, so their color strength is not comparable across categories.

To reproduce after PR #7 is present, run `python -m anomaly.pipeline.run_pipeline --dataset-root /path/to/mvtec-ad --output-root artifacts/full --image-size 256 --batch-size 8 --device mps --max-patches 2048 --projection-dim 256 --seed 42`. After PR #8 is present, render panels with `python -m scripts.reporting.visualize_anomalies --dataset-root /path/to/mvtec-ad --output-root artifacts/full --image-size 256 --device mps`. The local dataset, model banks, logs, and panels remain under ignored paths and are not part of this report.
