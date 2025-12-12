## ULS+ inference (nnU-Net v2 CLI) and evaluation

### 1) Set environment variables
```bash
export nnUNet_raw=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw
export nnUNet_results=/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_resultsv2
```

### 2) Run inference to shared storage
```bash
nnUNetv2_predict \
  -i /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/imagesTr \
  -o /data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/preds \
  -d Dataset090_ULS23_Combined \
  -c 3d_fullres_singlepass \
  -p nnUNetResEncUNetLPlans \
  -f all \
  -chk checkpoint_best.pth
```
If needed, you can switch to the latest checkpoint with `-chk checkpoint_latest.pth`.

Notes:
- The test dataset.json `numTraining` value does not affect inference.
- Input filenames must mirror training format (`*_0000.nii.gz`).

### 3) Run evaluation and write CSV (with progress + parallel workers)
The evaluation script prints tqdm progress bars and supports parallel processing via `--workers`.

Basic usage:
```bash
python3 nnunet_training/pipelines/eval_uls.py \
  --dataset-root /data/bodyct/.../nnUNet_raw/DatasetXXX_Test \
  --preds        /path/to/predictions_dir \
  --out          /path/to/output/uls_metrics.csv \
  --workers      12
```

The CSV contains per-type and overall Dice/Boundary IoU and agreement (mean pairwise Dice/Boundary IoU among normal/aug1/aug2).

### 4) Plot metrics (save PNGs)
Generate simple bar plots (Dice and Boundary IoU) from the CSV. Images are saved (no interactive display).
```bash
python3 nnunet_training/pipelines/plot_uls_metrics.py \
  --csv    /path/to/output/uls_metrics.csv \
  --outdir /path/to/output/plots   # optional; defaults to CSV folder
```
