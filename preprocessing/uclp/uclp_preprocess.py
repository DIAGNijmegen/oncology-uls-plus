"""Main entry point for UCLP preprocessing pipeline"""
import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime

from uclp.config import load_config
from uclp.io_utils import load_image, get_adapter
from uclp.mask_utils import get_connected_components, get_lesion_stats, check_lesion_truncation, binarize_mask
from uclp.cropper import generate_augmented_crops
from uclp.nnunet_writer import create_dataset_structure, generate_case_id, save_sample, write_dataset_json
import re


def extract_case_id(file_path):
    """Extract case identifier from file path.
    
    For most datasets, uses the filename stem.
    For WORC-style datasets (where image is always 'image.nii.gz'),
    uses the parent directory name as the case ID.
    """
    path = Path(file_path)
    stem = path.stem.replace('.nii', '').replace('.mha', '')
    
    # If filename is generic (like 'image'), use parent directory name
    if stem in ['image', 'scan', 'img']:
        return path.parent.name
    
    return stem


def process_case(img_path, label_path, config, output_paths, warnings_log):
    """Process a single CT-mask pair.
    
    Args:
        img_path: Path to CT scan
        label_path: Path to mask
        config: Configuration dict
        output_paths: Dict with output directory paths
        warnings_log: Open file handle for warnings
        
    Returns:
        int: Number of samples generated for this case
    """
    case_id = extract_case_id(img_path)
    
    try:
        # Load images
        image = load_image(str(img_path))
        mask = load_image(str(label_path))
        
        # Verify image and mask compatibility
        # For per-lesion labels, resample mask to image grid if needed
        per_lesion = bool(config.get('labels_are_per_lesion', False))
        if image.GetSize() != mask.GetSize():
            if per_lesion:
                # Resample mask to image grid (per-lesion masks are pre-cropped)
                import SimpleITK as sitk
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(image)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                mask = resampler.Execute(mask)
            else:
                print(f"  WARNING: Image and mask size mismatch for {case_id}, skipping")
                return 0
        
        # Filter labels and get connected components
        labels_to_keep = config.get('labels_to_keep', None)
        labels_to_remove = config.get('labels_to_remove', None)
        
        # Get filtered mask for cropping (only lesions, no organ)
        from uclp.mask_utils import filter_labels
        filtered_mask = filter_labels(mask, labels_to_keep, labels_to_remove)
        
        # Get connected components from filtered mask
        labeled_mask = get_connected_components(filtered_mask)
        lesion_stats = get_lesion_stats(labeled_mask)
        
        if not lesion_stats:
            print(f"  WARNING: No lesions found in {case_id}, skipping")
            return 0
        
        print(f"  Processing {case_id}: {len(lesion_stats)} lesion(s) found")
        
        # Get minimum lesion size threshold from config
        min_lesion_size_voxels = config.get('min_lesion_size_voxels', 100)
        
        # Optionally build lesion_id -> lesion_type mapping from per-case CSV
        lesion_type_mapping = {}
        append_lesion_type = bool(config.get('append_lesion_type', False))
        lesion_types_dir = config.get('lesion_types_dir', None)
        base_case_id = case_id.split('_')[0]
        if append_lesion_type and lesion_types_dir:
            from pathlib import Path as _Path
            csv_path = _Path(lesion_types_dir) / f"{base_case_id}.csv"
            if csv_path.exists():
                try:
                    import csv
                    with open(csv_path, newline='') as cf:
                        reader = csv.DictReader(cf)
                        if reader.fieldnames and 'lesion_id' in reader.fieldnames and 'lesion_type' in reader.fieldnames:
                            for row in reader:
                                try:
                                    lid = int(row['lesion_id'])
                                except Exception:
                                    continue
                                ltype = (row.get('lesion_type') or '').strip()
                                if ltype:
                                    lesion_type_mapping[lid] = ltype
                        else:
                            warnings_log.write(f"Missing columns in lesion types CSV for {base_case_id}: {csv_path}\n")
                except Exception as e:
                    warnings_log.write(f"Failed reading lesion types CSV for {base_case_id}: {e}\n")
            else:
                warnings_log.write(f"Lesion types CSV not found for {base_case_id}: {csv_path}\n")

        def _sanitize_type(val: str) -> str:
            s = val.lower().replace(' ', '-').replace('/', '-')
            return re.sub(r"[^a-z0-9_-]", "", s)

        # Determine per-lesion processing mode (each label file is a single lesion)
        per_lesion = bool(config.get('labels_are_per_lesion', False))
        parsed_lesion_id = None
        if per_lesion:
            m = re.search(r"_lesion_(\d+)_sample_1", Path(label_path).name)
            try:
                parsed_lesion_id = int(m.group(1)) if m else 0
            except Exception:
                parsed_lesion_id = 0

        # Build iteration list: either all components, or the single best component
        if per_lesion and lesion_stats:
            # Choose largest component if more than one exists unexpectedly
            best_idx = max(range(len(lesion_stats)), key=lambda i: lesion_stats[i]['volume_voxels'])
            loop_items = [(best_idx, lesion_stats[best_idx])]
        else:
            loop_items = list(enumerate(lesion_stats))

        # Process each lesion/component
        sample_count = 0
        skipped_lesions = 0
        for comp_idx, stats in loop_items:
            # Check lesion size
            lesion_size = stats['volume_voxels']
            if lesion_size < min_lesion_size_voxels:
                lesion_idx_for_name = parsed_lesion_id if per_lesion else comp_idx
                warning_msg = f"{config['name']}_{case_id}_lesion{lesion_idx_for_name}: too small ({lesion_size} voxels < {min_lesion_size_voxels} threshold), skipping\n"
                warnings_log.write(warning_msg)
                warnings_log.flush()
                skipped_lesions += 1
                continue
            
            # Create a mask with ONLY the current lesion (remove all peripheral lesions)
            import SimpleITK as sitk
            import numpy as np
            labeled_array = sitk.GetArrayFromImage(labeled_mask)
            single_lesion_array = (labeled_array == (comp_idx + 1)).astype(np.uint8)
            single_lesion_mask = sitk.GetImageFromArray(single_lesion_array)
            single_lesion_mask.CopyInformation(labeled_mask)
            
            # Determine original lesion label by sampling original mask at centroid; fallback to mode overlap
            orig_label = None
            if append_lesion_type and lesion_types_dir:
                try:
                    centroid_idx = mask.TransformPhysicalPointToIndex(stats['centroid_physical'])
                    size = mask.GetSize()
                    if all(0 <= i < s for i, s in zip(centroid_idx, size)):
                        orig_label = int(mask.GetPixel(centroid_idx))
                        if orig_label == 0:
                            raise ValueError('centroid at background')
                    else:
                        raise ValueError('centroid out of bounds')
                except Exception:
                    # Fallback: mode of labels within the component
                    import numpy as _np
                    orig_mask_np = sitk.GetArrayFromImage(mask)
                    labeled_array_local = sitk.GetArrayFromImage(labeled_mask)
                    single_lesion_array_local = (labeled_array_local == (lesion_idx + 1)).astype(_np.uint8)
                    vox = orig_mask_np[single_lesion_array_local > 0]
                    if vox.size > 0:
                        vals, counts = _np.unique(vox[vox > 0], return_counts=True)
                        if vals.size > 0:
                            orig_label = int(vals[_np.argmax(counts)])

            # Generate augmented crops using ONLY the single lesion mask
            lesion_idx_for_name = parsed_lesion_id if per_lesion else comp_idx
            seed_key = f"{Path(img_path).name}|lesion{lesion_idx_for_name}"
            crops = generate_augmented_crops(
                image,
                single_lesion_mask,  # Use single lesion mask only
                stats['centroid_physical'],
                config,
                seed_key=seed_key,
            )
            
            # Save each crop
            for aug_idx, (img_crop, mask_crop) in enumerate(crops):
                # Check truncation for first (centered) sample only
                # Check only the central lesion, not all lesions in the crop
                if aug_idx == 0:
                    # Crop the single lesion mask to check truncation
                    from uclp.cropper import compute_crop_region, extract_crop
                    region = compute_crop_region(
                        stats['centroid_physical'],
                        tuple(config['crop_size']),
                        image.GetSize(),
                        image.GetSpacing(),
                        (0, 0, 0),  # No offset for centered sample
                        image
                    )
                    single_lesion_crop = extract_crop(single_lesion_mask, region, is_mask=True)
                    
                    if check_lesion_truncation(single_lesion_crop, stats['bbox_voxel']):
                        warning_msg = f"{config['name']}_{case_id}_lesion{lesion_idx}: extends beyond VOI\n"
                        warnings_log.write(warning_msg)
                        warnings_log.flush()
                
                # Generate case ID with optional lesion type suffix
                lesion_type_for_name = None
                if append_lesion_type and lesion_types_dir:
                    if isinstance(orig_label, int) and orig_label in lesion_type_mapping:
                        lesion_type_for_name = _sanitize_type(lesion_type_mapping[orig_label])
                    else:
                        lesion_type_for_name = 'unknown'

                full_case_id = generate_case_id(
                    config['name'],
                    case_id,
                    lesion_idx_for_name,
                    aug_idx if aug_idx > 0 else None,
                    lesion_type=lesion_type_for_name,
                )
                
                # Binarize mask before saving (ensure 0 and 1 only)
                mask_crop_binary = binarize_mask(mask_crop)
                
                # Save sample
                save_sample(img_crop, mask_crop_binary, full_case_id, output_paths)
                sample_count += 1
        
        if skipped_lesions > 0:
            print(f"  Skipped {skipped_lesions} connected component(s) due to small size")
        
        return sample_count
        
    except Exception as e:
        print(f"  ERROR processing {case_id}: {e}")
        traceback.print_exc()
        return 0


def main():
    """Main pipeline orchestration"""
    parser = argparse.ArgumentParser(
        description='Universal CT Lesion Preprocessor - Convert CT datasets to nnU-Net format'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    try:
        # Load and validate configuration
        print("=" * 80)
        print("UCLP - Universal CT Lesion Preprocessor")
        print("=" * 80)
        print(f"\nLoading configuration from: {args.config}")
        
        config = load_config(args.config)
        
        print(f"\nConfiguration:")
        print(f"  Dataset: {config['name']} (ID: {config['id']})")
        print(f"  Adapter: {config['adapter']}")
        print(f"  Input images: {config['input_images']}")
        if 'input_labels' in config:
            print(f"  Input labels: {config['input_labels']}")
        if 'image_pattern' in config:
            print(f"  Image pattern: {config['image_pattern']}")
        if 'label_pattern' in config:
            print(f"  Label pattern: {config['label_pattern']}")
        print(f"  Output directory: {config['output_dir']}")
        print(f"  Crop size: {config['crop_size']} voxels")
        print(f"  Augmentations per lesion: {config['num_augmentations']}")
        print(f"  Body threshold: {config['body_threshold_hu']} HU")
        print(f"  Min lesion size: {config['min_lesion_size_voxels']} voxels")
        print(f"  Random seed: {config['random_seed']}")
        
        # Setup output structure
        print(f"\nCreating output directory structure...")
        output_paths = create_dataset_structure(
            config['output_dir'],
            config['id'],
            config['name']
        )
        print(f"  Output: {output_paths['dataset_root']}")
        
        # Match image-label pairs using adapter
        print(f"\nMatching image-label pairs using '{config['adapter']}' adapter...")
        adapter = get_adapter(config['adapter'])
        pairs = adapter(config)
        
        if not pairs:
            print("  ERROR: No matching image-label pairs found")
            return 1
        
        print(f"  Found {len(pairs)} matching pairs")
        
        # Set random seed once for reproducible augmentation across all cases
        random_seed = config.get('random_seed', None)
        if random_seed is not None:
            import numpy as np
            np.random.seed(random_seed)
            print(f"  Random seed set to: {random_seed}")
        
        # Open warnings log
        warnings_log = open(output_paths['warnings_log'], 'w')
        warnings_log.write(f"UCLP Preprocessing Warnings - {datetime.now().isoformat()}\n")
        warnings_log.write(f"Dataset: {config['name']} (ID: {config['id']})\n")
        warnings_log.write("=" * 80 + "\n\n")
        
        # Process each case
        print(f"\nProcessing cases...")
        print("-" * 80)
        
        total_samples = 0
        total_lesions = 0
        successful_cases = 0
        
        for idx, (img_path, label_path) in enumerate(pairs, 1):
            # Progress indicator
            progress = idx / len(pairs) * 100
            bar_length = 40
            filled = int(bar_length * idx / len(pairs))
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\r[{bar}] {progress:.1f}% ({idx}/{len(pairs)}) Processing {img_path.name}...", end='', flush=True)
            
            samples = process_case(img_path, label_path, config, output_paths, warnings_log)
            
            if samples > 0:
                total_samples += samples
                # Approximate lesion count (samples / num_augmentations)
                total_lesions += samples // config['num_augmentations']
                successful_cases += 1
        
        # Clear progress line and print completion
        print(f"\r{' ' * 120}\r", end='')  # Clear the line
        
        # Write summary statistics to warnings log
        warnings_log.write("\n" + "=" * 80 + "\n")
        warnings_log.write("PROCESSING SUMMARY\n")
        warnings_log.write("=" * 80 + "\n")
        warnings_log.write(f"Total cases processed: {successful_cases}/{len(pairs)}\n")
        warnings_log.write(f"Total lesions extracted: {total_lesions}\n")
        warnings_log.write(f"Total samples generated: {total_samples}\n")
        warnings_log.write(f"Processing completed: {datetime.now().isoformat()}\n")
        warnings_log.write("=" * 80 + "\n")
        
        warnings_log.close()
        
        # Generate dataset.json
        print("-" * 80)
        print(f"\nGenerating dataset.json...")
        write_dataset_json(output_paths['dataset_root'], total_samples, config['name'])
        
        # Log summary statistics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"  Total cases processed: {successful_cases}/{len(pairs)}")
        print(f"  Total lesions extracted: {total_lesions}")
        print(f"  Total samples generated: {total_samples}")
        print(f"  Processing time: {duration:.1f} seconds")
        print(f"  Output directory: {output_paths['dataset_root']}")
        print(f"  Warnings log: {output_paths['warnings_log']}")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
