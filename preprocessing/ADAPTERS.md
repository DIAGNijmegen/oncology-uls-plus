# Dataset Adapters Guide

UCLP uses a flexible adapter pattern to handle diverse dataset folder structures. This guide explains how to use and create adapters.

## Why Adapters?

Different medical imaging datasets have different organizational structures:
- Some have separate `images/` and `labels/` folders
- Others use `imagesTr/` and `labelsTr/` (MSD format)
- Some keep scans and masks in the same folder with naming conventions
- Custom datasets may have nested structures or special naming patterns

Adapters solve this by providing dataset-specific logic for finding and matching image-label pairs.

## Built-in Adapters

### 1. `default` Adapter

**Use case:** Images and labels in separate directories, matched by base filename.

**Configuration:**
```yaml
adapter: default  # or omit - this is the default
input_images: /data/dataset/images
input_labels: /data/dataset/labels
```

**Directory structure:**
```
/data/dataset/
├── images/
│   ├── patient001.nii.gz
│   ├── patient002.nii.gz
│   └── subfolder/
│       └── patient003.nii.gz
└── labels/
    ├── patient001.nii.gz
    ├── patient002.nii.gz
    └── subfolder/
        └── patient003.nii.gz
```

**Features:**
- Recursively searches both directories
- Matches files by base name (ignoring extensions)
- Supports nested folder structures
- Works with both .nii.gz and .mha formats

---

### 2. `msd` Adapter

**Use case:** Medical Segmentation Decathlon format.

**Configuration:**
```yaml
adapter: msd
input_images: /data/MSD/Task03_Liver/imagesTr
input_labels: /data/MSD/Task03_Liver/labelsTr
```

**Directory structure:**
```
/data/MSD/Task03_Liver/
├── imagesTr/
│   ├── liver_0.nii.gz
│   ├── liver_1.nii.gz
│   └── liver_2.nii.gz
└── labelsTr/
    ├── liver_0.nii.gz
    ├── liver_1.nii.gz
    └── liver_2.nii.gz
```

**Features:**
- Functionally identical to `default` adapter
- Named explicitly for MSD datasets
- Follows MSD naming conventions

---

### 3. `same_folder` Adapter

**Use case:** Images and masks in the same directory with different naming patterns.

**Configuration:**
```yaml
adapter: same_folder
input_images: /data/dataset
image_pattern: '*_CT.nii.gz'
label_pattern: '*_seg.nii.gz'
```

**Directory structure:**
```
/data/dataset/
├── case001_CT.nii.gz
├── case001_seg.nii.gz
├── case002_CT.nii.gz
├── case002_seg.nii.gz
└── subfolder/
    ├── case003_CT.nii.gz
    └── case003_seg.nii.gz
```

**Features:**
- Matches files using glob patterns
- Extracts case ID by removing pattern prefix/suffix
- `input_labels` is not required
- Supports nested folders
- Flexible pattern matching with wildcards

**Pattern examples:**
```yaml
# Suffix patterns
image_pattern: '*_scan.nii.gz'
label_pattern: '*_mask.nii.gz'

# Prefix patterns
image_pattern: 'CT_*.nii.gz'
label_pattern: 'SEG_*.nii.gz'

# Complex patterns
image_pattern: '*_t1_*.nii.gz'
label_pattern: '*_seg_*.nii.gz'
```

---

### 4. `worc` Adapter

**Use case:** WORC dataset with per-case folders, multiple lesions per case, and multi-rater annotations.

**Configuration:**
```yaml
adapter: worc
input_images: /data/worc
cohort_filter: CRLM              # Optional: filter by cohort prefix
rater_strategy: majority         # 'majority', 'union', or 'intersection'
authority_rater: RAD             # Tiebreaker for 50/50 splits
```

**Directory structure:**
```
/data/worc/
├── CRLM-001/
│   ├── image.nii.gz
│   ├── segmentation_lesion0_CNN.nii.gz
│   ├── segmentation_lesion0_RAD.nii.gz
│   ├── segmentation_lesion0_STUD1.nii.gz
│   ├── segmentation_lesion1_CNN.nii.gz
│   ├── segmentation_lesion1_RAD.nii.gz
│   └── segmentation_lesion1_STUD1.nii.gz
├── CRLM-002/
│   ├── image.nii.gz
│   └── ...
├── GIST-001/
│   ├── image.nii.gz
│   └── segmentation.nii.gz
└── ...
```

**Features:**
- Handles per-case folder structure
- Supports multiple lesions per case (each lesion gets separate segmentation files)
- Merges multi-rater annotations using voting strategies
- Filters by cohort prefix (e.g., process only CRLM, ignore GIST/Liver/Melanoma)
- Uses authority rater (RAD) as tiebreaker for 50/50 splits

**Rater merging strategies:**
- **majority** (recommended): Voxel is lesion if >50% of raters agree. For ties (50/50), uses authority rater (RAD)
- **union**: Voxel is lesion if ANY rater marked it (maximizes sensitivity)
- **intersection**: Voxel is lesion if ALL raters marked it (maximizes specificity)

**Filename parsing:**
- Images: `image.nii.gz` (one per case folder)
- Multi-lesion, multi-rater masks: `segmentation_lesion{N}_{RATER}.nii.gz` (CRLM)
- Single lesion masks: `segmentation.nii.gz` (GIST, Liver, etc.)

**Example configs:**
- CRLM (multi-rater): `configs/worc_crlm.yaml`
- GIST (single lesion): `configs/worc_gist.yaml`

**Cohort filtering:**
```yaml
# Process only CRLM cases
cohort_filter: CRLM

# Process only GIST cases
cohort_filter: GIST

# Process all cases (omit parameter)
# cohort_filter: null
```

---

### 5. `waw_tace` Adapter

**Use case:** WAW-TACE multiphase CT dataset with phase-specific tumor masks.

**Configuration:**
```yaml
adapter: waw_tace
input_images: /data/WAW-TACE/Images
input_labels: /data/WAW-TACE/Masks
phase: 1  # Optional: filter by phase (0=native, 1=arterial, 2=portal, 3=delayed)
```

**Directory structure:**
```
/data/WAW-TACE/
├── Images/
│   ├── 35/
│   │   ├── 35_0_scan.nii.gz  (native phase)
│   │   ├── 35_1_scan.nii.gz  (arterial phase)
│   │   ├── 35_2_scan.nii.gz  (portal phase)
│   │   └── 35_3_scan.nii.gz  (delayed phase)
│   └── 42/
│       ├── 42_0_scan.nii.gz
│       └── ...
└── Masks/
    ├── 35/
    │   ├── 35_1_0_tumor_seg.nrrd  (arterial, tumor 0)
    │   ├── 35_1_1_tumor_seg.nrrd  (arterial, tumor 1)
    │   └── 35_2_0_tumor_seg.nii.gz (portal, tumor 0)
    └── 42/
        └── ...
```

**Features:**
- Matches masks with correct phase scan based on filename
- Supports multiple tumors per patient per phase
- Handles both .nii.gz and .nrrd formats
- Optional phase filtering to process only specific phases
- Ensures tumor masks are matched to the phase where they were created

**Filename parsing:**
- Images: `{patient_id}_{phase}_scan.nii.gz`
- Masks: `{patient_id}_{phase}_{tumor_index}_tumor_seg.(nii.gz|nrrd)`

**Phase filtering:**
```yaml
# Process all phases (default)
adapter: waw_tace

# Process only arterial phase
adapter: waw_tace
phase: 1

# Process only portal phase  
adapter: waw_tace
phase: 2
```

## Creating Custom Adapters

### Step 1: Write the Adapter Function

Add your adapter to `uclp/io_utils.py`:

```python
def my_custom_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Custom adapter for my specific dataset format.
    
    Args:
        config: Configuration dict with dataset-specific parameters
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    # Example: Dataset with case folders containing 'scan.nii.gz' and 'mask.nii.gz'
    base_dir = Path(config['input_images'])
    pairs = []
    
    for case_folder in base_dir.iterdir():
        if case_folder.is_dir():
            scan_path = case_folder / 'scan.nii.gz'
            mask_path = case_folder / 'mask.nii.gz'
            
            if scan_path.exists() and mask_path.exists():
                pairs.append((scan_path, mask_path))
    
    return sorted(pairs)
```

### Step 2: Register the Adapter

Update the `get_adapter()` function in `uclp/io_utils.py`:

```python
def get_adapter(adapter_name: str) -> Callable:
    adapters = {
        'default': default_adapter,
        'msd': msd_adapter,
        'same_folder': same_folder_adapter,
        'my_custom': my_custom_adapter,  # Add your adapter here
    }
    
    if adapter_name not in adapters:
        raise ValueError(f"Unknown adapter: {adapter_name}. Available: {list(adapters.keys())}")
    
    return adapters[adapter_name]
```

### Step 3: Use Your Adapter

Create a config file:

```yaml
name: My_Dataset
id: 200
adapter: my_custom
input_images: /data/my_special_dataset
# Add any custom parameters your adapter needs
crop_size: [128, 128, 64]
num_augmentations: 3
```

## Adapter Best Practices

1. **Return sorted pairs** - Always sort the output for deterministic processing
2. **Handle missing files gracefully** - Check if files exist before adding to pairs
3. **Support recursive search** - Use `rglob()` for nested structures
4. **Extract clean case IDs** - Remove extensions and patterns properly
5. **Document expected structure** - Add clear docstrings with examples
6. **Validate config parameters** - Check for required custom parameters

## Example: Complex Custom Adapter

Here's a more complex example for a dataset with multiple raters:

```python
def multi_rater_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Adapter for datasets with multiple rater annotations.
    
    Expects config parameters:
        - input_images: Directory with CT scans
        - input_labels: Directory with rater folders
        - rater: Which rater to use (e.g., 'rater1', 'rater2', 'consensus')
    
    Directory structure:
        /data/
        ├── images/
        │   ├── case001.nii.gz
        │   └── case002.nii.gz
        └── labels/
            ├── rater1/
            │   ├── case001.nii.gz
            │   └── case002.nii.gz
            ├── rater2/
            │   ├── case001.nii.gz
            │   └── case002.nii.gz
            └── consensus/
                ├── case001.nii.gz
                └── case002.nii.gz
    """
    image_dir = Path(config['input_images'])
    label_dir = Path(config['input_labels']) / config['rater']
    
    if not label_dir.exists():
        raise ValueError(f"Rater directory not found: {label_dir}")
    
    # Find all images
    image_files = {}
    for ext in ['*.nii.gz', '*.mha']:
        for img_path in image_dir.rglob(ext):
            base_name = _extract_base_name(img_path)
            image_files[base_name] = img_path
    
    # Find corresponding labels for selected rater
    label_files = {}
    for ext in ['*.nii.gz', '*.mha']:
        for label_path in label_dir.rglob(ext):
            base_name = _extract_base_name(label_path)
            label_files[base_name] = label_path
    
    # Match pairs
    pairs = []
    for base_name, img_path in image_files.items():
        if base_name in label_files:
            pairs.append((img_path, label_files[base_name]))
    
    return sorted(pairs)
```

Usage:

```yaml
adapter: multi_rater
input_images: /data/images
input_labels: /data/labels
rater: consensus  # or 'rater1', 'rater2'
```

## Troubleshooting

### No matching pairs found

**Problem:** Adapter returns empty list

**Solutions:**
1. Check that paths in config exist
2. Verify file extensions (.nii.gz vs .nii)
3. Print debug info in your adapter to see what files are found
4. Check that base names match exactly (case-sensitive)

### Wrong files matched

**Problem:** Images matched with wrong labels

**Solutions:**
1. Verify pattern matching logic
2. Check for duplicate base names
3. Add more specific patterns
4. Use case ID extraction carefully

### Adapter not found

**Problem:** `ValueError: Unknown adapter: my_adapter`

**Solutions:**
1. Check spelling in config file
2. Verify adapter is registered in `get_adapter()`
3. Restart if code was modified

## Summary

Adapters provide a clean, extensible way to handle diverse dataset structures:

- **Use built-in adapters** for common formats (default, msd, same_folder)
- **Create custom adapters** for special dataset structures
- **Keep adapters simple** - just return matched (image, label) pairs
- **Document your adapters** - help future users understand the expected structure

The adapter pattern makes UCLP truly universal, capable of handling any dataset organization with minimal code changes.
