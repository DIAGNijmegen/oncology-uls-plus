## Training nnU-Net v2 (ResEnc-L) with **no resampling** (custom plans)

### 0) Merge multiple datasets into one nnU-Net dataset (required)
Before running nnU-Net, we merged all source datasets into **one** nnU-Net v2 dataset (`DatasetXXX_Name` in
`$nnUNet_raw`). This can be done:

- **Manually** (copy/link cases into one dataset), or
- **With a small Python script** (not provided here).

The merged dataset must follow nnU-Net v2 format (at minimum):
- `imagesTr/` and `labelsTr/` with consistent case ids and channel suffixes.
- a valid `dataset.json`

### 1) One-time setup
Make the custom resampling function available inside your **nnUNetv2 installation** so that the function name
`no_resampling_data_or_seg_to_shape` can be imported at runtime (we did this by merging/copying
`nnunetv2/preprocessing/resampling/custom_resampling.py` into the installed nnU-Net package).

Also ensure nnU-Net environment variables are set:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

Set your dataset id:

```bash
export DATASET_ID=90   # example
```

### 2) Extract fingerprint (no preprocessing yet)

```bash
nnUNetv2_extract_fingerprint -d ${DATASET_ID}
```

### 3) Create plans (ResEnc-L) with a custom plans identifier

```bash
nnUNetv2_plan_experiment \
  -d ${DATASET_ID} \
  -pl nnUNetPlannerResEncL \
  -overwrite_plans_name nnUNetResEncUNetLPlans_noresamp
```

This creates a plans file like:
`$nnUNet_preprocessed/DatasetXXX_.../nnUNetResEncUNetLPlans_noresamp.json`

### 4) Manually edit the plans JSON to disable resampling
In `nnUNetResEncUNetLPlans_noresamp.json`, change these keys to `no_resampling_data_or_seg_to_shape`:

- `resampling_fn_data`
- `resampling_fn_seg`
- `resampling_fn_probabilities`

### 5) Preprocess using the edited plans
Depending on your nnUNetv2 version, one of these works (both mean: preprocess with that plans identifier):

```bash
nnUNetv2_preprocess -d ${DATASET_ID} -plans_name nnUNetResEncUNetLPlans_noresamp
```

or:

```bash
nnUNetv2_preprocess -d ${DATASET_ID} -p nnUNetResEncUNetLPlans_noresamp
```

### 6) Train (ResEnc-L plans + no-resampling)

```bash
nnunetv2_wrapper nnUNetv2_train ${DATASET_ID} 3d_fullres all -p nnUNetResEncUNetLPlans_noresamp
```