"""I/O utilities for loading and saving medical images using SimpleITK"""

import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any
import fnmatch


def load_image(path: str) -> sitk.Image:
    """Load medical image using SimpleITK.
    
    Args:
        path: Path to image file (.nii.gz or .mha)
        
    Returns:
        SimpleITK Image object with metadata preserved
    """
    return sitk.ReadImage(str(path))


def save_image(image: sitk.Image, path: str, compress: bool = True) -> None:
    """Save medical image with compression enabled by default.
    
    Args:
        image: SimpleITK Image object
        path: Output file path
        compress: Enable compression (default: True)
    """
    sitk.WriteImage(image, str(path), compress)


def default_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Match CT scans with masks in separate directories by base filename.
    
    Args:
        config: Configuration dict with 'input_images' and 'input_labels' keys
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    image_dir = Path(config['input_images'])
    label_dir = Path(config['input_labels'])
    
    # Recursively find all image files
    image_extensions = ['*.nii.gz', '*.mha']
    image_files = {}
    for ext in image_extensions:
        for img_path in image_dir.rglob(ext):
            base_name = _extract_base_name(img_path)
            image_files[base_name] = img_path
    
    # Recursively find all label files
    label_files = {}
    for ext in image_extensions:
        for label_path in label_dir.rglob(ext):
            base_name = _extract_base_name(label_path)
            label_files[base_name] = label_path
    
    # Match pairs by base name
    pairs = []
    for base_name, img_path in image_files.items():
        if base_name in label_files:
            pairs.append((img_path, label_files[base_name]))
    
    return sorted(pairs)


def msd_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Medical Segmentation Decathlon format adapter.
    
    Expects imagesTr/ and labelsTr/ directories with matching filenames.
    This is functionally identical to default_adapter but named for clarity.
    
    Args:
        config: Configuration dict with 'input_images' and 'input_labels' keys
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    return default_adapter(config)


def same_folder_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Match images and masks in the same directory using filename patterns.
    
    Args:
        config: Configuration dict with:
            - 'input_images': Directory containing both images and masks
            - 'image_pattern': Glob pattern for images (e.g., '*_scan.nii.gz')
            - 'label_pattern': Glob pattern for masks (e.g., '*_mask.nii.gz')
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    input_dir = Path(config['input_images'])
    image_pattern = config.get('image_pattern', '*_scan.*')
    label_pattern = config.get('label_pattern', '*_mask.*')
    
    # Find all files matching image pattern
    image_files = {}
    for img_path in input_dir.rglob('*'):
        if img_path.is_file() and fnmatch.fnmatch(img_path.name, image_pattern):
            # Extract case ID by removing pattern suffix
            case_id = _extract_case_id_from_pattern(img_path.name, image_pattern)
            image_files[case_id] = img_path
    
    # Find all files matching label pattern
    label_files = {}
    for label_path in input_dir.rglob('*'):
        if label_path.is_file() and fnmatch.fnmatch(label_path.name, label_pattern):
            case_id = _extract_case_id_from_pattern(label_path.name, label_pattern)
            label_files[case_id] = label_path
    
    # Match pairs by case ID
    pairs = []
    for case_id, img_path in image_files.items():
        if case_id in label_files:
            pairs.append((img_path, label_files[case_id]))
    
    return sorted(pairs)


def _merge_tumor_masks(mask_paths: List[Path], output_path: Path, reference_image: 'sitk.Image' = None) -> None:
    """Merge multiple tumor mask files into a single combined mask.
    
    Args:
        mask_paths: List of paths to tumor mask files to merge
        output_path: Path where merged mask should be saved
        reference_image: Optional reference image to resample masks to (e.g., the CT scan)
    """
    import SimpleITK as sitk
    import numpy as np
    
    if not mask_paths:
        return
    
    # Load first mask as base
    first_mask = sitk.ReadImage(str(mask_paths[0]))
    
    # If reference image provided, resample to match it
    if reference_image is not None:
        reference_size = reference_image.GetSize()
        reference_spacing = reference_image.GetSpacing()
        reference_origin = reference_image.GetOrigin()
        reference_direction = reference_image.GetDirection()
        
        # Resample first mask to reference
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for masks
        resampler.SetDefaultPixelValue(0)
        combined_mask = resampler.Execute(first_mask)
    else:
        combined_mask = first_mask
    
    combined_array = sitk.GetArrayFromImage(combined_mask)
    
    # Merge additional masks
    for mask_path in mask_paths[1:]:
        mask = sitk.ReadImage(str(mask_path))
        
        # Resample to match combined mask if needed
        if reference_image is not None or mask.GetSize() != combined_mask.GetSize():
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(combined_mask)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            mask = resampler.Execute(mask)
        
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Union: any non-zero voxel in either mask becomes 1
        combined_array = np.logical_or(combined_array > 0, mask_array > 0).astype(np.uint8)
    
    # Create output image with merged mask
    merged_image = sitk.GetImageFromArray(combined_array)
    merged_image.CopyInformation(combined_mask)
    
    # Save merged mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(merged_image, str(output_path), useCompression=True)


def waw_tace_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """WAW-TACE dataset adapter for multiphase CT with phase-specific tumor masks.
    
    Dataset structure:
        WAW-TACE/
        ├── Images/
        │   └── {patient_id}/
        │       ├── {patient_id}_0_scan.nii.gz  (native)
        │       ├── {patient_id}_1_scan.nii.gz  (arterial)
        │       ├── {patient_id}_2_scan.nii.gz  (portal)
        │       └── {patient_id}_3_scan.nii.gz  (delayed)
        └── Masks/
            └── {patient_id}/
                ├── {patient_id}_0_0_tumor_seg.nii.gz
                ├── {patient_id}_1_0_tumor_seg.nrrd
                └── ...
    
    Args:
        config: Configuration dict with:
            - 'input_images': Path to Images/ directory
            - 'input_labels': Path to Masks/ directory
            - 'phase': Optional phase to process (0-3). If not specified, processes all phases.
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    images_dir = Path(config['input_images'])
    masks_dir = Path(config['input_labels'])
    target_phase = config.get('phase', None)  # None means all phases
    
    # Create temp directory for merged masks
    import tempfile
    temp_dir = Path(tempfile.mkdtemp(prefix='waw_tace_merged_'))
    
    pairs = []
    
    # Iterate through patient directories in Images/
    for patient_dir in sorted(images_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        
        patient_id = patient_dir.name
        
        # Find all scan files for this patient
        scan_files = {}
        for scan_path in patient_dir.glob('*_scan.nii.gz'):
            # Extract phase from filename: {patient_id}_{phase}_scan.nii.gz
            parts = scan_path.stem.replace('.nii', '').split('_')
            if len(parts) >= 2:
                phase = parts[-2]  # Phase is second-to-last part
                scan_files[phase] = scan_path
        
        # Find corresponding masks
        mask_dir = masks_dir / patient_id
        if not mask_dir.exists():
            continue
        
        # Group tumor masks by phase
        masks_by_phase = {}  # phase -> list of mask paths
        
        for mask_path in mask_dir.glob('*_tumor_seg.*'):
            # Extract patient_id, phase, and tumor_index from filename
            # Format: {patient_id}_{phase}_{tumor_index}_tumor_seg.(nii.gz|nrrd)
            filename = mask_path.name
            
            # Remove extension(s)
            if filename.endswith('.nii.gz'):
                base = filename[:-7]
            elif filename.endswith('.nrrd'):
                base = filename[:-5]
            elif filename.endswith('.nii'):
                base = filename[:-4]
            else:
                continue
            
            # Split and extract phase
            # Format: {patient_id}_{phase}_{tumor_index}_tumor_seg
            # After removing '_tumor_seg', we have: {patient_id}_{phase}_{tumor_index}
            if base.endswith('_tumor_seg'):
                base = base[:-10]  # Remove '_tumor_seg'
            else:
                continue  # Skip if doesn't match expected format
            
            parts = base.split('_')
            if len(parts) >= 3:  # At minimum: patient_id, phase, tumor_index
                phase = parts[-2]  # Phase is second from end
                
                # Skip if we're filtering by phase and this doesn't match
                if target_phase is not None and str(phase) != str(target_phase):
                    continue
                
                # Group masks by phase
                if phase not in masks_by_phase:
                    masks_by_phase[phase] = []
                masks_by_phase[phase].append(mask_path)
        
        # Merge masks for each phase and create pairs
        for phase, mask_paths in masks_by_phase.items():
            if phase not in scan_files:
                continue
            
            if len(mask_paths) == 1:
                # Only one mask, use it directly
                pairs.append((scan_files[phase], mask_paths[0]))
            else:
                # Multiple masks, merge them
                # Load reference scan to ensure masks are resampled to match
                import SimpleITK as sitk
                reference_scan = sitk.ReadImage(str(scan_files[phase]))
                
                merged_mask_path = temp_dir / f"{patient_id}_{phase}_merged_mask.nii.gz"
                _merge_tumor_masks(mask_paths, merged_mask_path, reference_scan)
                pairs.append((scan_files[phase], merged_mask_path))
    
    return sorted(pairs)


def _merge_multi_rater_masks(mask_paths: List[Path], output_path: Path, reference_image: 'sitk.Image', strategy: str = 'majority', authority_rater: str = 'RAD') -> None:
    """Merge multiple rater masks using voting strategy with authority tiebreaker.
    
    Args:
        mask_paths: List of paths to rater mask files
        output_path: Path where merged mask should be saved
        reference_image: Reference image to resample masks to
        strategy: 'majority' (≥50% agreement), 'union' (any rater), or 'intersection' (all raters)
        authority_rater: Rater name to use as tiebreaker (default: 'RAD')
    """
    import SimpleITK as sitk
    import numpy as np
    
    if not mask_paths:
        return
    
    # Load and resample all masks to reference
    mask_arrays = []
    authority_mask = None
    
    for mask_path in mask_paths:
        mask = sitk.ReadImage(str(mask_path))
        
        # Resample to match reference
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        mask = resampler.Execute(mask)
        
        mask_array = sitk.GetArrayFromImage(mask)
        binary_mask = (mask_array > 0).astype(np.uint8)
        mask_arrays.append(binary_mask)
        
        # Track authority rater mask for tiebreaking
        if authority_rater in str(mask_path):
            authority_mask = binary_mask
    
    # Stack and apply voting strategy
    stacked = np.stack(mask_arrays, axis=0)  # Shape: (num_raters, z, y, x)
    
    if strategy == 'majority':
        # Majority vote: voxel is lesion if >50% of raters agree
        vote_sum = np.sum(stacked, axis=0)
        num_raters = len(mask_arrays)
        
        # Clear majority (>50%)
        merged_array = (vote_sum > num_raters / 2.0).astype(np.uint8)
        
        # Handle ties with authority rater
        if authority_mask is not None and num_raters % 2 == 0:
            # For even number of raters, exact 50% split is a tie
            tie_mask = (vote_sum == num_raters / 2.0)
            merged_array[tie_mask] = authority_mask[tie_mask]
    elif strategy == 'union':
        # Union: voxel is lesion if ANY rater marked it
        merged_array = np.any(stacked, axis=0).astype(np.uint8)
    elif strategy == 'intersection':
        # Intersection: voxel is lesion if ALL raters marked it
        merged_array = np.all(stacked, axis=0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")
    
    # Create output image
    merged_image = sitk.GetImageFromArray(merged_array)
    merged_image.CopyInformation(reference_image)
    
    # Save merged mask
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(merged_image, str(output_path), useCompression=True)


def worc_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """WORC dataset adapter for per-case folders with multiple lesion segmentations.
    
    Dataset structure:
        worc/
        ├── CRLM-001/
        │   ├── image.nii.gz
        │   ├── segmentation_lesion0_CNN.nii.gz
        │   ├── segmentation_lesion0_PhD.nii.gz
        │   └── ...
        └── GIST-001/
            ├── image.nii.gz
            └── segmentation.nii.gz
    
    Args:
        config: Configuration dict with:
            - 'input_images': Path to worc/ directory
            - 'cohort_filter': Optional cohort prefix to filter (e.g., 'CRLM', 'GIST')
            - 'rater_strategy': 'majority' (default), 'union', or 'intersection'
            - 'preferred_rater': Optional specific rater to use (e.g., 'PhD')
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    import tempfile
    
    worc_dir = Path(config['input_images'])
    cohort_filter = config.get('cohort_filter', None)
    rater_strategy = config.get('rater_strategy', 'majority')
    preferred_rater = config.get('preferred_rater', None)
    
    # Create temp directory for merged masks
    temp_dir = Path(tempfile.mkdtemp(prefix='worc_merged_'))
    
    pairs = []
    
    # Iterate through case directories
    for case_dir in sorted(worc_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        
        # Filter by cohort if specified
        if cohort_filter and not case_dir.name.startswith(cohort_filter):
            continue
        
        case_id = case_dir.name
        
        # Find image file
        image_path = case_dir / 'image.nii.gz'
        if not image_path.exists():
            continue
        
        # Find all segmentation files
        seg_files = list(case_dir.glob('segmentation*.nii.gz'))
        if not seg_files:
            continue
        
        # Group segmentations by lesion index
        lesions_by_index = {}  # lesion_index -> list of mask paths
        
        for seg_path in seg_files:
            filename = seg_path.stem.replace('.nii', '')
            
            # Parse filename: segmentation_lesion{n} or segmentation_lesion{n}_{RATER}
            if 'lesion' in filename:
                parts = filename.split('_')
                lesion_part = [p for p in parts if p.startswith('lesion')]
                if lesion_part:
                    lesion_idx = lesion_part[0].replace('lesion', '')
                    
                    # Check if specific rater requested
                    if preferred_rater:
                        if preferred_rater in filename:
                            lesions_by_index[lesion_idx] = [seg_path]
                    else:
                        if lesion_idx not in lesions_by_index:
                            lesions_by_index[lesion_idx] = []
                        lesions_by_index[lesion_idx].append(seg_path)
            else:
                # Simple segmentation.nii.gz (e.g., GIST)
                lesions_by_index['0'] = [seg_path]
        
        # Process each lesion
        import SimpleITK as sitk
        reference_image = sitk.ReadImage(str(image_path))
        
        for lesion_idx, mask_paths in lesions_by_index.items():
            if len(mask_paths) == 1:
                # Single mask, use directly
                pairs.append((image_path, mask_paths[0]))
            else:
                # Multiple raters, merge using strategy
                merged_mask_path = temp_dir / f"{case_id}_lesion{lesion_idx}_merged.nii.gz"
                authority_rater = config.get('authority_rater', 'RAD')
                _merge_multi_rater_masks(mask_paths, merged_mask_path, reference_image, rater_strategy, authority_rater)
                pairs.append((image_path, merged_mask_path))
    
    return sorted(pairs)


def mswal_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """MSWAL dataset adapter for nnU-Net format with channel suffix mismatch.
    
    Images have _0000 suffix but labels don't.
    Example: MSWAL_0001_0000.nii.gz (image) matches MSWAL_0001.nii.gz (label)
    
    Args:
        config: Configuration dict with 'input_images' and 'input_labels' keys
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    image_dir = Path(config['input_images'])
    label_dir = Path(config['input_labels'])
    
    pairs = []
    
    # Find all image files
    for img_path in sorted(image_dir.glob('*.nii.gz')):
        # Extract case ID by removing _0000 suffix
        # MSWAL_0001_0000.nii.gz -> MSWAL_0001
        img_base = img_path.stem.replace('.nii', '')
        if img_base.endswith('_0000'):
            case_id = img_base[:-5]  # Remove '_0000'
            
            # Look for corresponding label file
            label_path = label_dir / f"{case_id}.nii.gz"
            if label_path.exists():
                pairs.append((img_path, label_path))
    
    return sorted(pairs)


def longitudinal_ct_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Longitudinal CT dataset adapter for baseline and followup scans.
    
    Dataset structure:
        Longitudinal-CT/
        ├── inputsTr/
        │   ├── {case_id}_BL_img_BL_img_00.nii.gz  (baseline image, primary scan)
        │   ├── {case_id}_BL_img_BL_img_01.nii.gz  (baseline image, secondary scan)
        │   ├── {case_id}_BL_mask_BL_img_00.nii.gz (baseline mask, primary)
        │   ├── {case_id}_BL_mask_BL_img_01.nii.gz (baseline mask, secondary)
        │   ├── {case_id}_FU_img_FU_img_00.nii.gz  (followup image, primary)
        │   └── ...
        └── targetsTr/
            ├── {case_id}_FU_mask_FU_img_00.nii.gz (followup mask, primary)
            ├── {case_id}_FU_mask_FU_img_01.nii.gz (followup mask, secondary)
            └── ...
    
    Filename format: {case_id}_{timepoint}_{type}_{timepoint}_{type}_{scan_type}.nii.gz
    - timepoint: BL (baseline) or FU (followup)
    - type: img (image) or mask (mask)
    - scan_type: 00 (primary/torso) or 01 (secondary/head)
    
    Args:
        config: Configuration dict with:
            - 'input_images': Path to Longitudinal-CT directory
            - 'timepoint': Optional filter ('BL' or 'FU')
            - 'scan_type': Optional filter ('00' or '01')
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    base_dir = Path(config['input_images'])
    inputs_dir = base_dir / 'inputsTr'
    targets_dir = base_dir / 'targetsTr'
    
    timepoint_filter = config.get('timepoint', None)
    scan_type_filter = config.get('scan_type', None)
    
    pairs = []
    
    # Find all image files in inputsTr
    for img_file in sorted(inputs_dir.glob('*_img_*_img_*.nii.gz')):
        # Parse filename: {case_id}_{timepoint}_img_{timepoint}_img_{scan_type}.nii.gz
        filename = img_file.stem.replace('.nii', '')
        parts = filename.split('_')
        
        if len(parts) >= 6:
            case_id = parts[0]
            timepoint = parts[1]  # BL or FU
            scan_type = parts[-1]  # 00 or 01
            
            # Apply filters
            if timepoint_filter and timepoint != timepoint_filter:
                continue
            if scan_type_filter and scan_type != scan_type_filter:
                continue
            
            # Construct mask filename
            mask_filename = f"{case_id}_{timepoint}_mask_{timepoint}_img_{scan_type}.nii.gz"
            
            # Check for mask in appropriate directory
            if timepoint == 'BL':
                mask_file = inputs_dir / mask_filename
            else:  # FU
                mask_file = targets_dir / mask_filename
            
            if mask_file.exists():
                pairs.append((img_file, mask_file))
    
    return sorted(pairs)


def _read_patient_ids(csv_path: str) -> set[str]:
    """Read a CSV file containing a column 'patient' with case IDs.
    
    Falls back to reading raw lines if header is missing.
    """
    import csv
    patient_ids: set[str] = set()
    with open(csv_path, newline='') as f:
        try:
            reader = csv.DictReader(f)
            if reader.fieldnames and 'patient' in reader.fieldnames:
                for row in reader:
                    val = (row.get('patient') or '').strip()
                    if val:
                        patient_ids.add(val)
                return patient_ids
        except Exception:
            pass
        # Fallback: rewind and read non-empty lines excluding header token
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line or line.lower() == 'patient':
                continue
            patient_ids.add(line)
    return patient_ids


def longitudinal_ct_test_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """Longitudinal-CT test adapter filtered by a CSV of case IDs.
    
    Expects config keys:
      - input_images: Path to Longitudinal-CT base folder
      - patient_list_csv: CSV file with header 'patient' listing held-out case_ids
      - Optional: timepoint (BL|FU), scan_type (00|01)
    """
    allowed_ids = _read_patient_ids(config['patient_list_csv'])

    # Reuse the base adapter to assemble pairs, then filter
    all_pairs = longitudinal_ct_adapter(config)

    def _base_case_id(p: Path) -> str:
        stem = p.stem.replace('.nii', '')
        return stem.split('_')[0]

    filtered = [(img, lbl) for img, lbl in all_pairs if _base_case_id(img) in allowed_ids]
    return sorted(filtered)

def cect_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """CECT dataset adapter for multiphase CT with separate ct_files and mask_files folders.
    
    Dataset structure:
        CECT/
        ├── ct_files/
        │   ├── P0001_ct_P.nii.gz
        │   ├── P0001_ct_C1.nii.gz
        │   ├── P0001_ct_C2.nii.gz
        │   ├── P0001_ct_C3.nii.gz
        │   └── ...
        └── mask_files/
            ├── P0001_mask_P.nii.gz
            ├── P0001_mask_C1.nii.gz
            ├── P0001_mask_C2.nii.gz
            ├── P0001_mask_C3.nii.gz
            └── ...
    
    Args:
        config: Configuration dict with:
            - 'input_images': Path to CECT/ct_files directory
            - 'input_labels': Path to CECT/mask_files directory
            - 'phase': Optional phase filter (e.g., 'C1', 'C2', 'C3', 'P')
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    ct_dir = Path(config['input_images'])
    mask_dir = Path(config['input_labels'])
    phase_filter = config.get('phase', None)
    
    pairs = []
    
    # Find all CT files
    for ct_file in sorted(ct_dir.glob('P*_ct_*.nii.gz')):
        # Parse filename: P{case_id}_ct_{phase}.nii.gz
        filename = ct_file.stem.replace('.nii', '')
        parts = filename.split('_')
        
        if len(parts) >= 3:
            case_id = parts[0]  # P0001
            phase = parts[2]     # P, C1, C2, or C3
            
            # Apply phase filter if specified
            if phase_filter and phase != phase_filter:
                continue
            
            # Construct corresponding mask filename
            mask_file = mask_dir / f"{case_id}_mask_{phase}.nii.gz"
            
            if mask_file.exists():
                pairs.append((ct_file, mask_file))
    
    return sorted(pairs)


def clm_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """CLM dataset adapter for nested case/study/images-masks structure.
    
    Dataset structure:
        CLM/nifti/
        ├── CRLM-CT-1001/
        │   └── study-{date}_{description}/
        │       ├── images/
        │       │   └── {case_id}_{date}_ct-{series}_na.nii.gz
        │       └── masks/
        │           └── {case_id}_{date}_tumor.nii.gz
        └── CRLM-CT-1002/
            └── ...
    
    Args:
        config: Configuration dict with 'input_images' pointing to CLM/nifti/ directory
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    clm_dir = Path(config['input_images'])
    pairs = []
    
    # Iterate through case directories (CRLM-CT-XXXX)
    for case_dir in sorted(clm_dir.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith('.'):
            continue
        
        # Iterate through study directories (study-{date}_{description})
        for study_dir in case_dir.iterdir():
            if not study_dir.is_dir() or study_dir.name.startswith('.'):
                continue
            
            images_dir = study_dir / 'images'
            masks_dir = study_dir / 'masks'
            
            if not images_dir.exists() or not masks_dir.exists():
                continue
            
            # Find image and mask files
            image_files = list(images_dir.glob('*.nii.gz'))
            mask_files = list(masks_dir.glob('*tumor*.nii.gz'))
            
            # Match pairs (should be 1 image and 1 mask per study)
            if len(image_files) == 1 and len(mask_files) == 1:
                pairs.append((image_files[0], mask_files[0]))
    
    return sorted(pairs)


def uls23_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """
    ULS23 dataset adapter for images/ and labels/ folder structure.
    
    This is essentially the same as the default adapter, but named specifically
    for ULS23 datasets for clarity.
    
    Dataset structure:
        dataset_folder/
        ├── images/
        │   ├── case_001.nii.gz
        │   └── case_002.nii.gz
        └── labels/
            ├── case_001.nii.gz
            └── case_002.nii.gz
    
    Args:
        config: Configuration dict with:
            - 'input_images': Path to images/ directory
            - 'input_labels': Path to labels/ directory
        
    Returns:
        List of (image_path, label_path) tuples for matched pairs
    """
    # ULS23 uses the same structure as default adapter
    return default_adapter(config)


def get_adapter(adapter_name: str) -> Callable:
    """Get adapter function by name.
    
    Args:
        adapter_name: Name of adapter ('default', 'msd', 'same_folder', 'waw_tace', 'worc', 'uls23', etc.)
        
    Returns:
        Adapter function that takes config and returns list of (image, label) pairs
        
    Raises:
        ValueError: If adapter name is not recognized
    """
    adapters = {
        'default': default_adapter,
        'msd': msd_adapter,
        'mswal': mswal_adapter,
        'same_folder': same_folder_adapter,
        'waw_tace': waw_tace_adapter,
        'worc': worc_adapter,
        'cect': cect_adapter,
        'longitudinal_ct': longitudinal_ct_adapter,
        'longitudinal_ct_test': longitudinal_ct_test_adapter,
        'clm': clm_adapter,
        'uls23': uls23_adapter,
        'lndb': msd_adapter,
        'uls_challenge': uls_challenge_adapter,
    }
    
    if adapter_name not in adapters:
        raise ValueError(f"Unknown adapter: {adapter_name}. Available adapters: {list(adapters.keys())}")
    
    return adapters[adapter_name]


def _extract_base_name(file_path: Path) -> str:
    """Extract base name from file path, handling .nii.gz extension."""
    name = file_path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    if name.endswith('.mha'):
        return name[:-4]
    return file_path.stem


def _extract_case_id_from_pattern(filename: str, pattern: str) -> str:
    """Extract case ID from filename using pattern.
    
    Assumes pattern has a wildcard (*) that represents the case ID.
    Example: 'case001_scan.nii.gz' with pattern '*_scan.nii.gz' -> 'case001'
    """
    # Simple extraction: remove pattern suffix/prefix
    # Find the * in pattern and extract corresponding part from filename
    if '*' not in pattern:
        return filename
    
    # Split pattern by *
    parts = pattern.split('*')
    case_id = filename
    
    # Remove prefix
    if parts[0]:
        case_id = case_id.replace(parts[0], '', 1)
    
    # Remove suffix
    if parts[-1]:
        if case_id.endswith(parts[-1]):
            case_id = case_id[:-len(parts[-1])]
    
    return case_id


def uls_challenge_adapter(config: Dict[str, Any]) -> List[Tuple[Path, Path]]:
    """ULS Challenge adapter: per-lesion labels, keep only sample_1.
    
    Dataset structure:
        images: /data/bodyct/experiments/nielsrocholl/ULS+/images
            - {case_id}.nii.gz
        labels: /uls_data/challenge/experiments/test_set/labelsTs
            - {case_id}_lesion_{NN}_sample_1.nii.gz
            - {case_id}_lesion_{NN}_sample_2.nii.gz  (ignored)
            - {case_id}_lesion_{NN}_sample_3.nii.gz  (ignored)

    Behavior:
        - Returns one (image, label) pair per lesion (uses only sample_1)
        - Does not merge masks
    """
    from pathlib import Path as _Path
    import re as _re

    images_dir = _Path(config['input_images'])
    labels_dir = _Path(config['input_labels'])

    def _normalize_case_id_str(s: str) -> str:
        # Strip common nnU-Net channel suffix
        return s[:-5] if s.endswith('_0000') else s

    def _image_case_id(p: Path) -> str:
        base = _extract_base_name(p)
        # If filename is generic, use parent folder as case id (matches preprocess logic)
        if base in ['image', 'scan', 'img']:
            base = p.parent.name
        return _normalize_case_id_str(base)

    # Build map of case_id -> image path
    image_map: Dict[str, Path] = {}
    for ext in ['*.nii.gz', '*.mha', '*.nii']:
        for img_path in images_dir.rglob(ext):
            case_id = _image_case_id(img_path)
            image_map[case_id] = img_path

    pairs: List[Tuple[Path, Path]] = []

    # Iterate label files and keep only sample_1
    label_files: List[Path] = []
    for ext in ['*.nii.gz', '*.nii']:
        label_files.extend(labels_dir.rglob(ext))

    for label_path in label_files:
        name = label_path.name
        if '_lesion_' not in name or '_sample_1' not in name:
            continue

        try:
            prefix = name[:name.index('_lesion_')]
        except ValueError:
            continue

        norm_prefix = _normalize_case_id_str(prefix)

        # First try direct path in images_dir with common extensions
        direct = None
        for ext in ('.nii.gz', '.nii', '.mha'):
            p = images_dir / f"{norm_prefix}{ext}"
            if p.exists():
                direct = p
                break

        img = direct if direct is not None else image_map.get(norm_prefix)
        if img is not None:
            pairs.append((img, label_path))

    if config.get('debug_pairs', False):
        print(f"  [uls_challenge] images: {len(image_map)}, labels considered: {len(label_files)}, pairs: {len(pairs)}")
        if pairs:
            for i, (im, lb) in enumerate(pairs[:5]):
                print(f"    pair[{i}]: {im.name} <-> {lb.name}")
        else:
            # Print a few sample label prefixes and whether they exist in image_map
            sample_labels = label_files[:5]
            print("    sample label prefixes (-> found image?):")
            for lb in sample_labels:
                nm = lb.name
                if '_lesion_' in nm and '_sample_1' in nm:
                    pref = nm[:nm.index('_lesion_')]
                    key = _normalize_case_id_str(pref)
                    print(f"      {pref} -> {'yes' if key in image_map else 'no'}")
            # Print a few image keys
            try:
                img_keys = list(image_map.keys())[:5]
                print("    sample image ids:")
                for k in img_keys:
                    print(f"      {k}")
            except Exception:
                pass

    return sorted(pairs)
