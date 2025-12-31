# ULS+ Evaluation

Evaluate predictions and generate metrics/plots.

## Prerequisites

- Predictions generated (see `training/inference.md`)
- Test dataset with ground truth labels organized in nnUNet format

## Setup

Set environment variables:

```bash
export nnUNet_raw=<path/to/raw/data>
export nnUNet_results=<path/to/model/results>
```

## Run Evaluation

### 1. Generate Metrics CSV

Evaluate predictions against ground truth and compute Dice/Boundary IoU metrics:

```bash
python3 evaluation/eval_uls.py \
  --dataset-root <path/to/test/dataset> \
  --preds        <path/to/predictions> \
  --out          <path/to/output/metrics.csv> \
  --workers      12
```

**Arguments:**
- `--dataset-root`: Test dataset root directory (default: Dataset401_Longitudinal_CT_Test_128)
- `--preds`: Directory containing prediction files
- `--out`: Output CSV path for metrics
- `--workers`: Number of parallel workers (default: 1)
- `--labels`: Optional custom labels directory (default: `dataset-root/labelsTr`)
- `--per-sample-csv`: Optional path to write per-sample Dice scores
- `--per-sample-agreement-csv`: Optional path to write per-sample agreement scores

**Output:**
The CSV contains per-lesion-type and overall metrics:
- Dice score and Boundary IoU (mean ± std)
- Agreement metrics (mean pairwise Dice/BIoU among normal/aug1/aug2 predictions)
- Sample counts per lesion type

### 2. Generate Plots

#### Single Dataset Plots

Generate bar plots for a single evaluation CSV:

```bash
python3 evaluation/plot_uls_metrics.py \
  --csv    <path/to/metrics.csv> \
  --outdir <path/to/output/plots>  # optional; defaults to CSV folder
```

**Output:** `uls_dice_by_type.png` and `uls_biou_by_type.png`

#### Comparison Plots (ULS vs ULS+)

Generate side-by-side comparison plots between two models:

```bash
python3 evaluation/plot_uls_combined.py \
  --csv-uls      <path/to/uls_metrics.csv> \
  --csv-uls-plus <path/to/uls_plus_metrics.csv> \
  --outdir       <path/to/output/plots>  # optional; defaults to CSV folder
```

**Output:**
- `combined_dice_by_type.png`
- `combined_biou_by_type.png`
- `combined_agreement_dice_by_type.png`
- `combined_agreement_biou_by_type.png`

## Notes

- Input prediction filenames must match label format (e.g., `*_128_updated_*` → `*_256_updated_*`)
- The evaluation script uses tqdm progress bars and supports parallel processing
- Agreement metrics require predictions with `_aug1` and `_aug2` variants
