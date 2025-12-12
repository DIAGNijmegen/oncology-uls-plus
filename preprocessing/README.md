# Universal CT Lesion Preprocessor (UCLP)

High-efficiency multi-dataset converter that transforms arbitrary CT datasets into VOIs in nnU-Net compliant format.

## Features

- Extract lesion-centered subvolumes from CT scans
- Generate augmented samples with off-center cropping
- Maintain complete traceability from processed samples to original data
- Ensure spatial consistency for medical imaging workflows
- Output nnU-Net compliant datasets ready for training

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.8
- SimpleITK ≥ 2.0 (all image operations)
- NumPy ≥ 1.20 (array operations)
- PyYAML ≥ 5.4 (configuration parsing)

## Usage

```bash
python uclp/uclp_preprocess.py --config configs/msd_liver.yaml
```

## Configuration

Create a YAML configuration file with the following parameters:

```yaml
name: MSD_Liver              # Dataset name
id: 31                       # Dataset ID
adapter: msd                 # Dataset adapter (optional, default: 'default')
input_images: /path/to/images
input_labels: /path/to/labels
labels_to_keep: [2]          # Optional: keep only specific labels (e.g., [2] for lesions)
labels_to_remove: [1]        # Optional: remove specific labels (e.g., [1] for organ)
output_dir: ./nnUNet_raw     # Optional, default: ./nnUNet_raw/
crop_size: [128, 128, 64]    # [x, y, z] in voxels
num_augmentations: 3         # Samples per lesion
 # offset_range_mm is deprecated and ignored in latest pipeline
random_seed: 42              # For reproducibility (optional, default: 42)
```

**Note:** All output masks are automatically binarized to 0 (background) and 1 (lesion) regardless of input label values.

See `configs/` directory for example configurations.

### Label Filtering

For datasets with multiple labels (e.g., organ + lesion), you can filter labels during preprocessing:

```yaml
# Keep only specific labels (e.g., label 2 for lesions in MSD datasets)
labels_to_keep: [2]

# Or remove specific labels (e.g., label 1 for organ mask)
labels_to_remove: [1]
```

**Important:** All output masks are automatically converted to binary format (0=background, 1=lesion), regardless of the input label values. This ensures compatibility with nnU-Net and consistent output across all datasets.

### Dataset Adapters

UCLP uses adapters to handle different dataset folder structures. Three built-in adapters are available:

#### 1. `default` - Separate Directories (Default)

For datasets with images and labels in separate directories, matched by base filename.

```yaml
adapter: default  # or omit this line
input_images: /data/dataset/images
input_labels: /data/dataset/labels
```

**Example structure:**
```
/data/dataset/
├── images/
│   ├── case001.nii.gz
│   └── case002.nii.gz
└── labels/
    ├── case001.nii.gz
    └── case002.nii.gz
```

#### 2. `msd` - Medical Segmentation Decathlon

Optimized for MSD format with `imagesTr/` and `labelsTr/` directories.

```yaml
adapter: msd
input_images: /data/MSD/Task03_Liver/imagesTr
input_labels: /data/MSD/Task03_Liver/labelsTr
```

**Example structure:**
```
/data/MSD/Task03_Liver/
├── imagesTr/
│   ├── liver_130.nii.gz
│   └── liver_131.nii.gz
└── labelsTr/
    ├── liver_130.nii.gz
    └── liver_131.nii.gz
```

#### 3. `same_folder` - Images and Masks in Same Directory

For datasets where images and masks are in the same directory with different naming patterns.

```yaml
adapter: same_folder
input_images: /data/dataset
image_pattern: '*_scan.nii.gz'   # Pattern for CT scans
label_pattern: '*_mask.nii.gz'   # Pattern for masks
```

**Example structure:**
```
/data/dataset/
├── case001_scan.nii.gz
├── case001_mask.nii.gz
├── case002_scan.nii.gz
└── case002_mask.nii.gz
```

**Note:** When using `same_folder` adapter, `input_labels` is not required. The adapter extracts case IDs by removing the pattern prefix/suffix.

#### 4. `waw_tace` - Multiphase CT Dataset

For WAW-TACE style datasets with multiphase CT scans and phase-specific tumor masks.

```yaml
adapter: waw_tace
input_images: /data/WAW-TACE/Images
input_labels: /data/WAW-TACE/Masks
phase: 1  # Optional: process only arterial phase (0=native, 1=arterial, 2=portal, 3=delayed)
```

**Example structure:**
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
        └── 35_1_1_tumor_seg.nrrd  (arterial, tumor 1)
```

**Features:**
- Automatically matches masks with correct phase scan
- Supports multiple tumors per patient per phase
- Handles both .nii.gz and .nrrd formats
- Optional phase filtering

### Creating Custom Adapters

You can easily add custom adapters for your specific dataset structure. Add a new function to `uclp/io_utils.py`:

```python
def my_custom_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Custom adapter for my dataset format.
    
    Args:
        config: Configuration dict
        
    Returns:
        List of (image_path, label_path) tuples
    """
    # Your custom logic to find and match image-label pairs
    pairs = []
    # ... implementation ...
    return sorted(pairs)
```

Then register it in the `get_adapter()` function:

```python
adapters = {
    'default': default_adapter,
    'msd': msd_adapter,
    'same_folder': same_folder_adapter,
    'my_custom': my_custom_adapter,  # Add your adapter
}
```

Use it in your config:

```yaml
adapter: my_custom
```

## Output Format

Generates nnU-Net compliant datasets:

```
nnUNet_raw/DatasetXXX_Name/
├── dataset.json
├── preprocessing_warnings.log
├── imagesTr/
│   ├── {SOURCE}_{CASEID}_lesion{IDX}_0000.nii.gz
│   ├── {SOURCE}_{CASEID}_lesion{IDX}_aug1_0000.nii.gz
│   └── ...
└── labelsTr/
    ├── {SOURCE}_{CASEID}_lesion{IDX}.nii.gz
    ├── {SOURCE}_{CASEID}_lesion{IDX}_aug1.nii.gz
    └── ...
```

## Quick Start Examples

### Example 1: Test with Provided Example Files
```bash
# Quick test using the included example CT scan and mask
python uclp/uclp_preprocess.py --config configs/test_example.yaml
```

### Example 2: MSD Format Dataset
```bash
# Process Medical Segmentation Decathlon datasets
python uclp/uclp_preprocess.py --config configs/msd_liver.yaml
```

### Example 3: Standard Separate Directories
```bash
# Process datasets with separate image/label folders
python uclp/uclp_preprocess.py --config configs/worc_melanoma.yaml

# Process WORC CRLM with multi-rater annotation merging
python uclp/uclp_preprocess.py --config configs/worc_crlm.yaml

# Process WORC GIST (single lesion per case)
python uclp/uclp_preprocess.py --config configs/worc_gist.yaml
```

## Project Structure

```
uclp/
├── config.py           # YAML configuration loading and validation
├── io_utils.py         # SimpleITK-based image I/O with dataset adapters
├── mask_utils.py       # Connected component analysis
├── cropper.py          # Lesion-centered cropping with metadata preservation
├── nnunet_writer.py    # dataset.json generation and file writing
└── uclp_preprocess.py  # Main entry point

configs/
├── test_example.yaml           # Quick test with provided example files (same_folder adapter)
├── msd_liver.yaml              # MSD Liver dataset example (msd adapter)
├── mswal.yaml                  # MSWAL multi-label abdominal lesions (msd adapter)
├── worc_melanoma.yaml          # WORC dataset example (default adapter)
├── worc_crlm.yaml              # WORC CRLM with multi-rater merging (worc adapter)
├── worc_gist.yaml              # WORC GIST single-lesion cases (worc adapter)
├── cect.yaml                   # CECT multiphase CT dataset (cect adapter)
├── longitudinal_ct.yaml        # Longitudinal CT baseline/followup (longitudinal_ct adapter)
├── clm.yaml                    # CLM dataset with nested structure (clm adapter)
└── waw_tace.yaml               # WAW-TACE multiphase CT example (waw_tace adapter)

example_ct_files/
├── scan.nii.gz         # Example CT scan for testing
└── mask.nii.gz         # Example lesion mask for testing
```
