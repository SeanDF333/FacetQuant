# FacetQuant

Per-crystal instance segmentation and 2D projected area measurement for protein crystallization micrographs. The first model is a **Mask R-CNN ResNet50-FPN v2** initialized from COCO weights and fine-tuned on the [CHiMP Detector Datasets](https://zenodo.org/records/11110373). Each predicted crystal mask is measured independently; the output includes a per-crystal CSV, mask archives, overlays, and an area histogram.

## Quick inference

```bash
python -m pip install -e .
facetquant predict --input /path/to/images --weights weights/facetquant_fp16.pth --out results
```

For physical units, pass `--um-per-pixel 0.5` (replace `0.5` with the calibration of the **original** input image). Without calibration, areas are in px². `--score-threshold` defaults to 0.6; `--exclude-border` removes partial objects touching an image boundary.

The output folder contains:

- `crystals.csv`: one row per detected crystal, including score, area, and bounding box.
- `masks/*.npz`: one Boolean mask per detected instance and its score.
- `overlays/*.png`: predicted crystal masks over the input images.
- `area_distribution.png`: a histogram of all detected crystal areas.

The model was trained on protein crystallization sitting-drop images. Images from a different microscope or crystallization process require a labeled local validation set and may require fine-tuning. The CHiMP annotations cover crystals selected as interesting or mountable by experts, so counts and area distributions relative to these labels do **not** guarantee all crystals in a field of view were measured.

## Dataset and published splits

Download [Zenodo record 11110373](https://zenodo.org/records/11110373) and unzip `chimp_detector_datasets.zip`. The archive contains JPEG images, `.npz` files with `masks` and `class_labels`, and four published split CSVs. The dataset is **not** committed to this repository.

| Subset | Images | Published training CSV | Published validation CSV |
| --- | ---: | ---: | ---: |
| VMXi | 237 | **189** | **48** |
| XChem | 350 | **280** | **70** |
| Combined | 587 | **469** | **118** |

The [CHiMP paper](https://journals.iucr.org/d/issues/2024/10/00/nj5327/index.html) describes VMXi as 190/47, but the actual CSVs in the Zenodo archive contain 189/48. The code follows the CSV files. The paper and archive provide **validation** splits, not a separate instance-mask test set. The paper's 632/949-image “test sets” are for **image-level classification**, not per-crystal mask or area evaluation.

## Training

Use a GPU and the official [TorchVision Mask R-CNN v2 COCO checkpoint](https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth):

```bash
python -m pip install -e '.[eval]'
facetquant train \
  --data /path/to/unzipped/chimp_detector_datasets \
  --coco-weights /path/to/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth \
  --sources vmxi,xchem --epochs 8 --max-size 768 \
  --out runs/baseline
facetquant evaluate \
  --data /path/to/unzipped/chimp_detector_datasets \
  --weights runs/baseline/best_fp16.pth \
  --sources vmxi,xchem --max-size 768 \
  --out runs/baseline/validation_metrics.json
```

`train` saves the minimum-validation-loss weights in half precision and a full training checkpoint. The first baseline trains only the **crystal** class, even when the `.npz` also contains a drop mask. It uses the published training CSVs and evaluates on the published validation CSVs.

## Metrics

- **Mask AP**: COCO mean average precision for instance masks, averaged over IoU thresholds 0.50, 0.55, ..., 0.95. `AP50` and `AP75` fix the mask IoU threshold at 0.50 and 0.75. Box AP uses bounding boxes instead.
- **Count MAE**: mean absolute difference between the number of predicted crystals above score 0.6 and the number of annotated crystal instances, per image.
- **Matched area median APE**: median of `|predicted area − annotated area| / annotated area`, for one-to-one crystal matches with mask IoU ≥ 0.5. This excludes missed and unmatched predicted crystals, so read it together with AP and count MAE.
- **Size-specific AP**: COCO small, medium, and large object definitions. Small crystals are particularly difficult in these images.

All area metrics are 2D image projection areas. A single image cannot uniquely reveal the area of a crystal portion hidden behind another crystal. Metrics also inherit the dataset's selective annotation policy.

The [original CHiMP work](https://github.com/OllyK/chimp-tools) trained a Mask R-CNN and reported VMXi crystal mask AP 0.288 and XChem two-class mask AP 0.386 on their validation sets. Our combined, crystal-only model is a different experiment; its validation score should not be treated as directly comparable to the paper's XChem score.

## Citation

King, O. N. F. et al. (2024), “CHiMP: deep-learning tools trained on protein crystallization micrographs to enable automation of experiments,” *Acta Crystallographica D* 80, 744–764. [DOI: 10.1107/S2059798324009276](https://doi.org/10.1107/S2059798324009276).
