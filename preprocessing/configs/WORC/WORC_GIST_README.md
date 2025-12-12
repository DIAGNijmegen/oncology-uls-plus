# WORC GIST Preprocessing Guide

## Overview

This guide explains how to preprocess the WORC GIST (Gastrointestinal Stromal Tumors) dataset. GIST cases are simpler than CRLM:
- Per-case folder structure
- Single image per case
- Single segmentation file per case (no multiple raters or lesions)

## Configuration

Edit `configs/worc_gist.yaml` to match your dataset location:

```yaml
name: WORC_GIST
id: 38

adapter: worc
input_images: /path/to/worc  # UPDATE THIS PATH

# WORC-specific parameters
cohort_filter: GIST              # Only process GIST cases

# Crop and augmentation parameters
crop_size: [128, 128, 64]
num_augmentations: 3
offset_range_mm: 10.0
offset_axes: ['z']

random_seed: 42
```

## Expected Dataset Structure

```
worc/
├── GIST-001/
│   ├── image.nii.gz
│   └── segmentation.nii.gz
├── GIST-002/
│   ├── image.nii.gz
│   └── segmentation.nii.gz
├── GIST-003/
│   └── ...
└── ...
```

**Important notes:**
- Each case has its own folder (e.g., `GIST-001`)
- One `image.nii.gz` per case
- One `segmentation.nii.gz` per case (binary mask)
- No multi-rater annotations (unlike CRLM)
- Typically one lesion per case, but if multiple lesions exist in the mask, they will be processed as separate connected components

## Running the Preprocessing

1. **Update the config file** with your dataset path:
   ```bash
   # Edit configs/worc_gist.yaml
   # Change: input_images: /path/to/worc
   ```

2. **Run the preprocessing**:
   ```bash
   python uclp/uclp_preprocess.py --config configs/worc_gist.yaml
   ```

3. **Output location**:
   ```
   nnUNet_raw/Dataset038_WORC_GIST/
   ├── dataset.json
   ├── imagesTr/
   │   ├── GIST-001_lesion0_0000.nii.gz
   │   ├── GIST-001_lesion0_aug1_0000.nii.gz
   │   ├── GIST-002_lesion0_0000.nii.gz
   │   └── ...
   └── labelsTr/
       ├── GIST-001_lesion0.nii.gz
       ├── GIST-001_lesion0_aug1.nii.gz
       ├── GIST-002_lesion0.nii.gz
       └── ...
   ```

## Differences from CRLM

| Feature | CRLM | GIST |
|---------|------|------|
| Raters | Multiple (CNN, RAD, STUD1, STUD2) | Single |
| Lesions per case | Multiple (separate files) | Typically one |
| Segmentation filename | `segmentation_lesion{N}_{RATER}.nii.gz` | `segmentation.nii.gz` |
| Rater merging | Yes (majority voting + RAD tiebreaker) | No (not needed) |
| Complexity | High | Low |

## Processing Other WORC Cohorts

The WORC dataset contains multiple cohorts. To process others:

### Liver Metastases
```yaml
name: WORC_Liver
id: 39
cohort_filter: Liver
```

### Melanoma
```yaml
name: WORC_Melanoma
id: 40
cohort_filter: Melanoma
```

Or use the existing `configs/worc_melanoma.yaml` which uses the `default` adapter (assumes separate scans/segmentations folders).

## Troubleshooting

### No cases found
- Verify the `input_images` path points to the worc directory
- Check that case folders start with "GIST" (case-sensitive)
- Ensure each case folder contains both `image.nii.gz` and `segmentation.nii.gz`

### Multiple lesions detected
- If a GIST case has multiple disconnected lesions in the segmentation, they will be processed separately
- Each lesion gets its own index: `GIST-001_lesion0`, `GIST-001_lesion1`, etc.
- This is expected behavior and maintains consistency with other datasets

### Wrong cohort processed
- Double-check the `cohort_filter` parameter matches your folder names exactly
- Case-sensitive: `GIST` ≠ `gist`

## Technical Details

### Adapter Behavior
The `worc` adapter automatically detects the segmentation file type:
- If filename contains `lesion` → multi-lesion CRLM format
- Otherwise → simple single-file format (GIST)

### Connected Components
Even though GIST typically has one lesion per case, the pipeline:
1. Runs connected component analysis on the segmentation
2. Extracts each connected component as a separate lesion
3. Generates augmented samples for each lesion independently

This ensures consistency across all datasets and handles edge cases where multiple lesions exist.

### Spatial Consistency
- All crops maintain proper spatial metadata (origin, spacing, direction)
- Cropped regions align perfectly with original scans in 3D Slicer
- Origin is recalculated for each crop to ensure physical coordinate accuracy

## Summary

The WORC GIST preprocessing:
- ✅ Handles per-case folder structure
- ✅ Processes single segmentation file per case
- ✅ Filters to GIST cohort only (ignores CRLM, Liver, Melanoma)
- ✅ Handles multiple lesions if present in segmentation
- ✅ Generates augmented samples per lesion
- ✅ Outputs nnU-Net compatible dataset
- ✅ Maintains full traceability (case ID in filenames)
- ✅ Simpler than CRLM (no multi-rater merging needed)
