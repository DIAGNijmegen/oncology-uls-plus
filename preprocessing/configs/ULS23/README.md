# ULS23 Dataset Configurations

This directory contains configuration files for preprocessing the ULS23 (Universal Lesion Segmentation 2023) datasets.

## Dataset Overview

ULS23 contains 8 different CT lesion segmentation datasets organized into two categories:

### Fully Annotated (5 datasets)
Complete annotations for all lesions in each scan.

| Dataset | ID | Description | Config File |
|---------|----|-----------|----|
| KITS21 | 40 | Kidney tumors | `uls23_kits21.yaml` |
| LIDC-IDRI | 41 | Lung nodules | `uls23_lidc_idri.yaml` |
| LiTS | 42 | Liver tumors | `uls23_lits.yaml` |
| NIH_LN_ABD | 43 | Abdominal lymph nodes | `uls23_nih_ln_abd.yaml` |
| NIH_LN_MED | 44 | Mediastinal lymph nodes | `uls23_nih_ln_med.yaml` |

### Novel Data (3 datasets)
New datasets with single lesion annotations per scan.

| Dataset | ID | Description | Config File |
|---------|----|-----------|----|
| DeepLesion3D | 45 | Multi-organ lesions | `uls23_deeplesion3d.yaml` |
| Radboudumc Bone | 46 | Bone lesions | `uls23_radboudumc_bone.yaml` |
| Radboudumc Pancreas | 47 | Pancreatic lesions | `uls23_radboudumc_pancreas.yaml` |

## Dataset Structure

All ULS23 datasets follow the same simple structure:

```
dataset_folder/
├── images/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── labels/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

## Usage

Process each dataset individually:

```bash
# Fully Annotated Datasets
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_kits21.yaml
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_lidc_idri.yaml
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_lits.yaml
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_nih_ln_abd.yaml
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_nih_ln_med.yaml

# Novel Data Datasets
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_deeplesion3d.yaml
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_radboudumc_bone.yaml
python uclp/uclp_preprocess.py --config configs/ULS23/uls23_radboudumc_pancreas.yaml
```

## Configuration Parameters

All configs use standard parameters:

- **Crop size**: 128×128×64 voxels
- **Augmentations**: 3 per lesion (1 centered + 2 augmented)
- **Offset range**: 10mm spherical sampling
- **Min lesion size**: 100 voxels
- **Random seed**: 42

## Output

Each dataset will be processed into nnU-Net format:

```
nnUNet_raw/
├── Dataset040_ULS23_KITS21/
├── Dataset041_ULS23_LIDC_IDRI/
├── Dataset042_ULS23_LiTS/
├── Dataset043_ULS23_NIH_LN_ABD/
├── Dataset044_ULS23_NIH_LN_MED/
├── Dataset045_ULS23_DeepLesion3D/
├── Dataset046_ULS23_Radboudumc_Bone/
└── Dataset047_ULS23_Radboudumc_Pancreas/
```

Each dataset folder contains:
- `imagesTr/` - Cropped CT volumes
- `labelsTr/` - Corresponding lesion masks
- `dataset.json` - nnU-Net metadata
- `preprocessing_warnings.log` - Processing warnings and statistics

## Notes

- Each label contains exactly 1 connected component (1 lesion)
- Labels are binary (0=background, 1=lesion)
- Small lesions (<100 voxels) are automatically filtered out
- All datasets use the `uls23` adapter (simple images/labels matching)

## Batch Processing

To process all ULS23 datasets:

```bash
# Process all fully annotated
for config in configs/ULS23/uls23_{kits21,lidc_idri,lits,nih_ln_abd,nih_ln_med}.yaml; do
    python uclp/uclp_preprocess.py --config "$config"
done

# Process all novel data
for config in configs/ULS23/uls23_{deeplesion3d,radboudumc_bone,radboudumc_pancreas}.yaml; do
    python uclp/uclp_preprocess.py --config "$config"
done
```
