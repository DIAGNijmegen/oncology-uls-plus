# Configuration Examples

This directory contains example configuration files demonstrating different dataset adapters.

## Available Configs

### 1. `test_example.yaml` - Quick Test
**Adapter:** `same_folder`  
**Purpose:** Test the pipeline with provided example files  
**Use case:** Images and masks in the same directory with different naming patterns

```bash
python uclp/uclp_preprocess.py --config configs/test_example.yaml
```

**Dataset structure:**
```
example_ct_files/
├── scan.nii.gz
└── mask.nii.gz
```

---

### 2. `msd_liver.yaml` - Medical Segmentation Decathlon
**Adapter:** `msd`  
**Purpose:** Process MSD format datasets  
**Use case:** Medical Segmentation Decathlon with imagesTr/labelsTr folders

```bash
python uclp/uclp_preprocess.py --config configs/msd_liver.yaml
```

**Dataset structure:**
```
/data/MSD/Task03_Liver/
├── imagesTr/
│   ├── liver_0.nii.gz
│   └── liver_1.nii.gz
└── labelsTr/
    ├── liver_0.nii.gz
    └── liver_1.nii.gz
```

---

### 3. `worc_melanoma.yaml` - Standard Separate Directories
**Adapter:** `default`  
**Purpose:** Process datasets with separate image/label directories  
**Use case:** Standard structure with scans and segmentations in different folders

```bash
python uclp/uclp_preprocess.py --config configs/worc_melanoma.yaml
```

**Dataset structure:**
```
/data/WORC/Melanoma/
├── scans/
│   ├── patient001.nii.gz
│   └── patient002.nii.gz
└── segmentations/
    ├── patient001.nii.gz
    └── patient002.nii.gz
```

---

### 4. `waw_tace.yaml` - Multiphase CT Dataset
**Adapter:** `waw_tace`  
**Purpose:** Process WAW-TACE multiphase CT dataset with phase-specific tumor masks  
**Use case:** Multiphase CT scans with tumor masks created in specific phases

```bash
python uclp/uclp_preprocess.py --config configs/waw_tace.yaml
```

**Dataset structure:**
```
/data/WAW-TACE/
├── Images/
│   └── 35/
│       ├── 35_0_scan.nii.gz  (native)
│       ├── 35_1_scan.nii.gz  (arterial)
│       ├── 35_2_scan.nii.gz  (portal)
│       └── 35_3_scan.nii.gz  (delayed)
└── Masks/
    └── 35/
        ├── 35_1_0_tumor_seg.nrrd  (arterial, tumor 0)
        ├── 35_1_1_tumor_seg.nrrd  (arterial, tumor 1)
        └── 35_2_0_tumor_seg.nii.gz (portal, tumor 0)
```

**Special features:**
- Automatically matches masks with correct phase scan
- Supports multiple tumors per patient per phase
- Handles both .nii.gz and .nrrd formats
- Optional phase filtering (process only specific phase)

---

## Creating Your Own Config

1. Copy one of the example configs that matches your dataset structure
2. Update the following fields:
   - `name`: Your dataset name
   - `id`: Unique dataset ID (3-digit number)
   - `adapter`: Choose appropriate adapter (default, msd, or same_folder)
   - `input_images`: Path to your CT scans
   - `input_labels`: Path to your masks (not needed for same_folder adapter)
   - `crop_size`: Adjust based on your lesion sizes
   - `num_augmentations`: Number of samples per lesion
   - `offset_range_mm`: Deprecated and ignored in latest pipeline

3. Run the pipeline:
```bash
python uclp/uclp_preprocess.py --config configs/your_config.yaml
```

## Adapter Selection Guide

| Dataset Structure | Adapter | Config Example |
|------------------|---------|----------------|
| Separate `images/` and `labels/` folders | `default` | `worc_melanoma.yaml` |
| MSD format with `imagesTr/` and `labelsTr/` | `msd` | `msd_liver.yaml` |
| Same folder with naming patterns | `same_folder` | `test_example.yaml` |
| Multiphase CT with phase-specific masks | `waw_tace` | `waw_tace.yaml` |

For more details on adapters, see [ADAPTERS.md](../ADAPTERS.md) in the root directory.
