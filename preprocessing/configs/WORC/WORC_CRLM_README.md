# WORC CRLM Preprocessing Guide

## Overview

This guide explains how to preprocess the WORC CRLM (Colorectal Liver Metastases) dataset, which has unique characteristics:
- Per-case folder structure
- Multiple lesions per case (each with separate segmentation files)
- Multiple raters per lesion (CNN, RAD, STUD1, STUD2, etc.)

## Multi-Rater Annotation Strategy

The WORC adapter implements **majority voting with RAD (Radiologist) as tiebreaker** to produce high-quality merged annotations:

### How it works:
1. **Majority voting**: For each voxel, the label that appears in the majority of rater masks is selected
2. **RAD tiebreaker**: When there's a 50/50 split (e.g., 2 raters say lesion, 2 say background), the RAD annotation is used as the authoritative source
3. **Rationale**: 
   - Reduces individual rater errors and biases
   - Leverages expert radiologist knowledge for ambiguous cases
   - More robust than single-rater or simple union/intersection

### Alternative strategies:
- **union**: Include voxel if ANY rater marked it (maximizes sensitivity, may include false positives)
- **intersection**: Include voxel if ALL raters marked it (maximizes specificity, may miss true boundaries)

## Configuration

Edit `configs/worc_crlm.yaml` to match your dataset location:

```yaml
name: WORC_CRLM
id: 37

adapter: worc
input_images: /path/to/worc  # UPDATE THIS PATH

# WORC-specific parameters
cohort_filter: CRLM              # Only process CRLM cases
rater_strategy: majority         # Use majority voting
authority_rater: RAD             # Use RAD as tiebreaker

# Crop and augmentation parameters
crop_size: [128, 128, 64]
num_augmentations: 5
offset_range_mm: 8.0
offset_axes: ['x', 'y', 'z']

random_seed: 42
```

## Expected Dataset Structure

```
worc/
├── CRLM-001/
│   ├── image.nii.gz
│   ├── segmentation_lesion0_CNN.nii.gz
│   ├── segmentation_lesion0_RAD.nii.gz
│   ├── segmentation_lesion0_STUD1.nii.gz
│   ├── segmentation_lesion0_STUD2.nii.gz
│   ├── segmentation_lesion1_CNN.nii.gz
│   ├── segmentation_lesion1_RAD.nii.gz
│   └── ...
├── CRLM-002/
│   ├── image.nii.gz
│   └── ...
└── ...
```

**Important notes:**
- Each case has its own folder (e.g., `CRLM-001`)
- One `image.nii.gz` per case
- Multiple lesions per case: `segmentation_lesion0_*.nii.gz`, `segmentation_lesion1_*.nii.gz`, etc.
- Multiple raters per lesion: `*_CNN.nii.gz`, `*_RAD.nii.gz`, `*_STUD1.nii.gz`, etc.
- The adapter will merge all raters for each lesion using the specified strategy

## Running the Preprocessing

1. **Update the config file** with your dataset path:
   ```bash
   # Edit configs/worc_crlm.yaml
   # Change: input_images: /path/to/worc
   ```

2. **Run the preprocessing**:
   ```bash
   python uclp/uclp_preprocess.py --config configs/worc_crlm.yaml
   ```

3. **Output location**:
   ```
   nnUNet_raw/Dataset037_WORC_CRLM/
   ├── dataset.json
   ├── imagesTr/
   │   ├── CRLM-001_lesion0_0000.nii.gz
   │   ├── CRLM-001_lesion0_aug1_0000.nii.gz
   │   ├── CRLM-001_lesion1_0000.nii.gz
   │   └── ...
   └── labelsTr/
       ├── CRLM-001_lesion0.nii.gz
       ├── CRLM-001_lesion0_aug1.nii.gz
       ├── CRLM-001_lesion1.nii.gz
       └── ...
   ```

## Processing Other WORC Cohorts

The WORC dataset contains multiple cohorts (CRLM, GIST, Liver, Melanoma). To process other cohorts:

1. **Create a new config file** (e.g., `configs/worc_gist.yaml`)
2. **Update the cohort filter**:
   ```yaml
   name: WORC_GIST
   id: 38
   cohort_filter: GIST  # Change this
   ```
3. **Adjust parameters** as needed (GIST may have different lesion sizes, single rater, etc.)

## Troubleshooting

### No cases found
- Verify the `input_images` path points to the worc directory
- Check that case folders start with "CRLM" (case-sensitive)
- Ensure each case folder contains `image.nii.gz`

### Missing raters
- The adapter handles missing raters gracefully
- If RAD is missing, majority voting still works (no tiebreaker)
- Check filenames match pattern: `segmentation_lesion{N}_{RATER}.nii.gz`

### Single-rater cases
- If a lesion has only one rater, that mask is used directly (no merging)
- This is normal and expected for some cases

## Technical Details

### Rater Merging Process
1. Load all rater masks for a lesion
2. Resample to match reference image (if needed)
3. Convert to binary masks (>0 = lesion)
4. Stack masks and compute vote sum per voxel
5. Apply majority threshold (>50% agreement)
6. For ties, use RAD mask as tiebreaker
7. Save merged mask to temporary directory
8. Use merged mask for cropping

### Spatial Consistency
- All merged masks maintain proper spatial metadata (origin, spacing, direction)
- Cropped regions will align perfectly with original scans in 3D Slicer
- Origin is recalculated for each crop to ensure physical coordinate accuracy

## Summary

The WORC CRLM preprocessing:
- ✅ Handles per-case folder structure
- ✅ Processes multiple lesions per case independently
- ✅ Merges multi-rater annotations using majority voting + RAD tiebreaker
- ✅ Filters to CRLM cohort only (ignores GIST, Liver, Melanoma)
- ✅ Generates augmented samples per lesion
- ✅ Outputs nnU-Net compatible dataset
- ✅ Maintains full traceability (case ID + lesion index in filenames)
