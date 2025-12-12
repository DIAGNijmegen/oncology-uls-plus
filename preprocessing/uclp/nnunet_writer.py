import json
from pathlib import Path
import SimpleITK as sitk


def create_dataset_structure(output_dir, dataset_id, dataset_name):
    """Create nnU-Net folder hierarchy.
    
    Args:
        output_dir: Base output directory (e.g., './nnUNet_raw')
        dataset_id: Three-digit dataset ID (e.g., 31)
        dataset_name: Dataset name (e.g., 'MSD_Liver')
    
    Returns:
        dict: Paths to created directories
    """
    output_dir = Path(output_dir)
    dataset_folder = output_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    
    dataset_folder.mkdir(parents=True, exist_ok=True)
    images_tr.mkdir(exist_ok=True)
    labels_tr.mkdir(exist_ok=True)
    
    return {
        'dataset_root': dataset_folder,
        'images_tr': images_tr,
        'labels_tr': labels_tr,
        'warnings_log': dataset_folder / 'preprocessing_warnings.log'
    }


def generate_case_id(source_name, case_id, lesion_idx, aug_idx=None, lesion_type: str | None = None):
    """Generate traceable filename following naming convention.
    
    Format: {SOURCE}_{CASEID}_lesion{IDX}_aug{N}
    First sample (centered) omits _aug0 for cleaner naming.
    
    Args:
        source_name: Dataset source name (e.g., 'MSD_Liver')
        case_id: Original case identifier (e.g., '130')
        lesion_idx: Lesion index within the case (0-based)
        aug_idx: Augmentation index (None for centered sample, 1+ for augmented)
    
    Returns:
        str: Case identifier for filename
    """
    base = f"{source_name}_{case_id}_lesion{lesion_idx}"
    if lesion_type:
        base += f"_type-{lesion_type}"
    
    if aug_idx is not None and aug_idx > 0:
        return f"{base}_aug{aug_idx}"
    
    return base


def write_dataset_json(output_dir, num_samples, dataset_name):
    """Generate nnU-Net compliant dataset.json.
    
    Args:
        output_dir: Dataset root directory
        num_samples: Total number of training samples
        dataset_name: Name of the dataset
    """
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": num_samples,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    
    output_path = Path(output_dir) / "dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)


def save_sample(image, label, case_id, output_paths):
    """Save image/label pair with correct nnU-Net naming.
    
    Args:
        image: SimpleITK image (cropped CT)
        label: SimpleITK image (cropped mask)
        case_id: Case identifier from generate_case_id()
        output_paths: Dict with 'images_tr' and 'labels_tr' paths
    """
    image_filename = f"{case_id}_0000.nii.gz"
    label_filename = f"{case_id}.nii.gz"
    
    image_path = output_paths['images_tr'] / image_filename
    label_path = output_paths['labels_tr'] / label_filename
    
    sitk.WriteImage(image, str(image_path), useCompression=True)
    sitk.WriteImage(label, str(label_path), useCompression=True)
