import SimpleITK as sitk
import numpy as np


def filter_labels(mask, labels_to_keep=None, labels_to_remove=None):
    """
    Filter mask to keep or remove specific label values.
    
    Args:
        mask: SimpleITK.Image mask with integer labels
        labels_to_keep: List of label values to keep (e.g., [2] to keep only label 2)
        labels_to_remove: List of label values to remove (e.g., [1] to remove label 1)
        
    Returns:
        SimpleITK.Image with filtered labels, binarized to 0 and 1
    """
    mask_array = sitk.GetArrayFromImage(mask)
    filtered_array = np.zeros_like(mask_array, dtype=np.uint8)
    
    if labels_to_keep is not None:
        # Keep only specified labels
        for label in labels_to_keep:
            filtered_array[mask_array == label] = 1
    elif labels_to_remove is not None:
        # Remove specified labels, keep everything else
        filtered_array[mask_array > 0] = 1
        for label in labels_to_remove:
            filtered_array[mask_array == label] = 0
    else:
        # No filtering, just binarize
        filtered_array[mask_array > 0] = 1
    
    # Convert back to SimpleITK image with same metadata
    filtered_mask = sitk.GetImageFromArray(filtered_array)
    filtered_mask.CopyInformation(mask)
    
    return filtered_mask


def get_connected_components(mask):
    """
    Label connected components in a mask using 26-connectivity.
    Assumes mask is already filtered/binarized.
    
    Args:
        mask: SimpleITK.Image binary mask
        
    Returns:
        SimpleITK.Image with labeled connected components
    """
    # Ensure mask is binary
    binary_mask = binarize_mask(mask)
    
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)  # 26-connectivity for 3D
    labeled_mask = cc_filter.Execute(binary_mask)
    return labeled_mask


def get_lesion_stats(labeled_mask):
    """
    Extract statistics for each labeled lesion including bounding boxes and centroids.
    
    Args:
        labeled_mask: SimpleITK.Image with labeled connected components
        
    Returns:
        List of dicts containing lesion statistics in physical coordinates
    """
    stats_filter = sitk.LabelShapeStatisticsImageFilter()
    stats_filter.Execute(labeled_mask)
    
    lesion_stats = []
    for label in stats_filter.GetLabels():
        if label == 0:  # Skip background
            continue
            
        # Get bounding box in voxel indices
        bbox_voxel = stats_filter.GetBoundingBox(label)  # [x_start, y_start, z_start, x_size, y_size, z_size]
        
        # Convert to min/max indices
        x_start, y_start, z_start = bbox_voxel[0], bbox_voxel[1], bbox_voxel[2]
        x_size, y_size, z_size = bbox_voxel[3], bbox_voxel[4], bbox_voxel[5]
        
        # Convert bounding box corners to physical coordinates
        min_corner_physical = labeled_mask.TransformIndexToPhysicalPoint((x_start, y_start, z_start))
        max_corner_physical = labeled_mask.TransformIndexToPhysicalPoint(
            (x_start + x_size - 1, y_start + y_size - 1, z_start + z_size - 1)
        )
        
        # Get centroid in physical coordinates
        centroid_physical = stats_filter.GetCentroid(label)  # Already in physical coordinates
        
        # Get volume
        volume_mm3 = stats_filter.GetPhysicalSize(label)
        volume_voxels = stats_filter.GetNumberOfPixels(label)
        
        lesion_stats.append({
            'label': label,
            'centroid_physical': centroid_physical,  # (x, y, z) in mm
            'bbox_physical': (min_corner_physical, max_corner_physical),  # ((x_min, y_min, z_min), (x_max, y_max, z_max)) in mm
            'bbox_voxel': bbox_voxel,  # Keep voxel bbox for convenience
            'volume_mm3': volume_mm3,
            'volume_voxels': volume_voxels
        })
    
    return lesion_stats


def binarize_mask(mask):
    """
    Convert mask to binary (0 and 1 only).
    
    Args:
        mask: SimpleITK.Image mask with any integer values
        
    Returns:
        SimpleITK.Image with binary values (0=background, 1=lesion)
    """
    binary_mask = mask > 0
    binary_mask = sitk.Cast(binary_mask, sitk.sitkUInt8)
    return binary_mask


def check_lesion_truncation(lesion_mask, crop_region):
    """
    Check if the central lesion extends to the boundaries of the crop region.
    
    Args:
        lesion_mask: SimpleITK.Image of the cropped mask
        crop_region: tuple of (start_index, size) in voxel coordinates
        
    Returns:
        bool: True if lesion touches crop boundaries, False otherwise
    """
    # Get the mask as numpy array (z, y, x order)
    mask_array = sitk.GetArrayFromImage(lesion_mask)
    
    # Get the size of the crop
    size_z, size_y, size_x = mask_array.shape
    
    # Check if any non-zero voxels touch the boundaries
    # Check z boundaries (first and last slices)
    if np.any(mask_array[0, :, :] > 0) or np.any(mask_array[-1, :, :] > 0):
        return True
    
    # Check y boundaries (first and last rows)
    if np.any(mask_array[:, 0, :] > 0) or np.any(mask_array[:, -1, :] > 0):
        return True
    
    # Check x boundaries (first and last columns)
    if np.any(mask_array[:, :, 0] > 0) or np.any(mask_array[:, :, -1] > 0):
        return True
    
    return False
