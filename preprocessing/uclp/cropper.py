import SimpleITK as sitk
import numpy as np
import hashlib
from numpy.random import default_rng


def is_inside_body(image, point_physical, hu_threshold=-500):
    """
    Check if a physical point is inside the body using HU values.
    
    Args:
        image: SimpleITK.Image CT scan
        point_physical: tuple (x, y, z) in physical coordinates (mm)
        hu_threshold: HU threshold for body detection (default: -500)
        
    Returns:
        bool: True if point is inside body, False otherwise
    """
    try:
        # Convert physical coordinates to voxel indices
        voxel_index = image.TransformPhysicalPointToIndex(point_physical)
        
        # Check if indices are within image bounds
        size = image.GetSize()
        if not all(0 <= idx < s for idx, s in zip(voxel_index, size)):
            return False
        
        # Get HU value at this voxel
        hu_value = image.GetPixel(voxel_index)
        
        # Body tissue has HU > -500 (air/outside is ~-1000)
        return hu_value > hu_threshold
        
    except Exception:
        # If transformation fails, assume outside body
        return False


def sample_point_in_sphere(radius):
    """
    Sample a random point uniformly within a sphere of given radius.
    
    Args:
        radius: Maximum distance from origin (mm)
        
    Returns:
        tuple: (x, y, z) offset vector in mm
    """
    # Sample direction uniformly on unit sphere
    direction = np.random.randn(3)
    direction = direction / np.linalg.norm(direction)
    
    # Sample distance uniformly from [0, radius]
    distance = np.random.uniform(0, radius)
    
    # Return offset vector
    return tuple(direction * distance)


def sample_point_from_lesion_mask(lesion_mask):
    """
    Sample a random point from within the lesion volume.
    
    Args:
        lesion_mask: SimpleITK.Image binary mask of the lesion
        
    Returns:
        tuple: (x, y, z) physical coordinates (mm) of random point in lesion,
               or None if lesion is empty
    """
    # Get lesion voxel coordinates
    mask_array = sitk.GetArrayFromImage(lesion_mask)
    lesion_voxels = np.argwhere(mask_array > 0)  # Returns (z, y, x) indices
    
    if len(lesion_voxels) == 0:
        return None
    
    # Randomly select one voxel from the lesion
    random_idx = np.random.randint(0, len(lesion_voxels))
    voxel_zyx = lesion_voxels[random_idx]
    
    # Convert from numpy (z, y, x) to SimpleITK (x, y, z) order
    voxel_xyz = (int(voxel_zyx[2]), int(voxel_zyx[1]), int(voxel_zyx[0]))
    
    # Convert voxel indices to physical coordinates
    physical_point = lesion_mask.TransformIndexToPhysicalPoint(voxel_xyz)
    
    return physical_point


def find_valid_offset(image, centroid_physical, offset_range_mm, hu_threshold=-500, max_attempts=50):
    """
    Find a valid offset that keeps the crop center inside the body.
    
    Uses rejection sampling to find a random point on a sphere around the
    lesion centroid that is guaranteed to be inside the body.
    
    Args:
        image: SimpleITK.Image CT scan
        centroid_physical: tuple (x, y, z) lesion centroid in physical coordinates (mm)
        offset_range_mm: Maximum offset distance (mm)
        hu_threshold: HU threshold for body detection (default: -500)
        max_attempts: Maximum rejection sampling attempts (default: 50)
        
    Returns:
        tuple: (x, y, z) offset vector in mm, or (0, 0, 0) if no valid point found
    """
    for attempt in range(max_attempts):
        # Sample random point in sphere
        offset = sample_point_in_sphere(offset_range_mm)
        
        # Calculate new centroid
        new_centroid = tuple(c + o for c, o in zip(centroid_physical, offset))
        
        # Check if new centroid is inside body
        if is_inside_body(image, new_centroid, hu_threshold):
            return offset
    
    # Fallback: return zero offset (centered crop)
    return (0.0, 0.0, 0.0)


def compute_crop_region(lesion_centroid, crop_size, image_size, spacing, offset=(0, 0, 0), image=None):
    """
    Calculate crop boundaries centered on lesion with optional offset.
    
    Args:
        lesion_centroid: tuple (x, y, z) in physical coordinates (mm)
        crop_size: tuple (x, y, z) in voxels
        image_size: tuple (x, y, z) in voxels
        spacing: tuple (x, y, z) spacing in mm/voxel
        offset: tuple (x, y, z) offset in physical coordinates (mm)
        image: SimpleITK.Image (optional) for proper coordinate transformation
        
    Returns:
        tuple: (start_index, size) where start_index is (x, y, z) in voxels
    """
    # Apply offset to centroid (both in physical coordinates)
    offset_centroid = tuple(c + o for c, o in zip(lesion_centroid, offset))
    
    # Convert offset centroid to voxel indices
    # Use SimpleITK's transformation if image is provided, otherwise use simple division
    if image is not None:
        centroid_voxel = image.TransformPhysicalPointToIndex(offset_centroid)
    else:
        # Fallback: simple conversion (assumes origin at 0,0,0 and identity direction)
        centroid_voxel = tuple(int(round(c / s)) for c, s in zip(offset_centroid, spacing))
    
    # Calculate crop start indices (centered on lesion)
    # Allow negative indices - we'll pad later if needed
    start_index = tuple(c - cs // 2 for c, cs in zip(centroid_voxel, crop_size))
    
    # Return the desired crop region (may extend beyond image boundaries)
    # Padding will be applied later to ensure lesion stays centered
    return (start_index, crop_size)


def extract_crop(image, region, is_mask=False):
    """
    Extract subvolume with padding if crop extends beyond image boundaries.
    
    Ensures the lesion stays centered by padding with appropriate values:
    - CT images: -1000 HU (air)
    - Masks: 0 (background)
    
    Args:
        image: SimpleITK.Image to crop
        region: tuple (start_index, size) in voxels, both in (x, y, z) order
                start_index can be negative (will pad)
        is_mask: bool, True if this is a mask (pad with 0), False for CT image (pad with -1000)
        
    Returns:
        SimpleITK.Image cropped and padded volume with updated metadata
    """
    start_index, crop_size = region
    image_size = image.GetSize()
    
    # Calculate padding needed (if start_index is negative or crop extends beyond)
    pad_lower = [max(0, -s) for s in start_index]
    pad_upper = [max(0, (s + cs) - img_s) for s, cs, img_s in zip(start_index, crop_size, image_size)]
    
    # Adjust start_index to valid range for extraction
    adjusted_start = [max(0, s) for s in start_index]
    
    # Calculate actual extraction size (what we can extract from the image)
    # This is the overlap between the desired crop and the actual image
    extract_size = []
    for img_s, adj_s, cs, pl, pu in zip(image_size, adjusted_start, crop_size, pad_lower, pad_upper):
        # Maximum we can extract from this position
        available = img_s - adj_s
        # What we need (crop size minus padding)
        needed = cs - pl - pu
        # Take the minimum
        extract_size.append(max(0, min(available, needed)))
    
    # Convert vector images to scalar if needed (e.g., RGB to grayscale)
    if image.GetNumberOfComponentsPerPixel() > 1:
        # Vector image (RGB, multi-channel) - convert to scalar
        # Use VectorIndexSelectionCast to extract first channel
        cast_filter = sitk.VectorIndexSelectionCastImageFilter()
        cast_filter.SetIndex(0)  # Extract first channel
        image = cast_filter.Execute(image)
    
    # Extract the valid region (only if there's something to extract)
    if all(s > 0 for s in extract_size):
        extract_filter = sitk.ExtractImageFilter()
        extract_filter.SetSize(extract_size)
        extract_filter.SetIndex(adjusted_start)
        cropped = extract_filter.Execute(image)
    else:
        # Create empty image if nothing to extract
        cropped = sitk.Image(extract_size, image.GetPixelID())
        cropped.SetSpacing(image.GetSpacing())
        cropped.SetDirection(image.GetDirection())
    
    # Pad if necessary
    if any(p > 0 for p in pad_lower + pad_upper):
        # Choose padding value based on image type
        if is_mask:
            pad_value = 0.0  # Background for masks
        else:
            pad_value = -1000.0  # Air for CT images
        
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound(pad_lower)
        pad_filter.SetPadUpperBound(pad_upper)
        pad_filter.SetConstant(pad_value)
        cropped = pad_filter.Execute(cropped)
    
    # Update origin to reflect physical position of crop (using original start_index)
    # New origin = old origin + direction_matrix @ (start_index * spacing)
    original_origin = np.array(image.GetOrigin())
    direction_matrix = np.array(image.GetDirection()).reshape(3, 3)
    spacing = np.array(image.GetSpacing())
    start_physical = np.array(start_index) * spacing
    
    new_origin = original_origin + direction_matrix @ start_physical
    
    cropped.SetOrigin(tuple(new_origin))
    cropped.SetSpacing(image.GetSpacing())
    cropped.SetDirection(image.GetDirection())
    
    return cropped


def _derive_seed(seed_material: str) -> int:
    """
    Derive a 64-bit seed from arbitrary string material using SHA-256.
    """
    digest = hashlib.sha256(seed_material.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little', signed=False)


def generate_augmented_crops(image, mask, lesion_centroid, config, seed_key: str | None = None):
    """
    Generate N augmented samples with random spherical offsets.
    
    Uses spherical sampling to generate offsets uniformly within a sphere
    around the lesion centroid, ensuring crop centers stay inside the body.
    
    Args:
        image: SimpleITK.Image CT scan
        mask: SimpleITK.Image lesion mask
        lesion_centroid: tuple (x, y, z) in physical coordinates (mm)
        config: dict with keys: crop_size, num_augmentations,
                body_threshold_hu (optional), max_sampling_attempts (optional)
        
    Returns:
        list of tuples: [(image_crop, mask_crop), ...] for each augmentation
    """
    crop_size = tuple(config['crop_size'])  # (x, y, z)
    num_augmentations = config['num_augmentations']
    hu_threshold = config.get('body_threshold_hu', -500)
    
    # Get image properties
    image_size = image.GetSize()  # (x, y, z)
    spacing = image.GetSpacing()  # (x, y, z)
    
    crops = []
    
    # Precompute lesion voxel indices for deterministic sampling
    mask_array = sitk.GetArrayFromImage(mask)
    lesion_voxels = np.argwhere(mask_array > 0)  # (z, y, x)

    for aug_idx in range(num_augmentations):
        # First sample is always centered on lesion centroid
        if aug_idx == 0:
            crop_center = lesion_centroid
        else:
            # Choose a random point from lesion voxels; deterministic if seed_key provided
            if lesion_voxels.size == 0:
                random_point = None
            elif seed_key is not None:
                rng = default_rng(_derive_seed(f"{seed_key}|aug{aug_idx}"))
                vidx = int(rng.integers(0, len(lesion_voxels)))
                z, y, x = lesion_voxels[vidx]
                random_point = mask.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
            else:
                random_point = sample_point_from_lesion_mask(mask)

            if random_point is None:
                crop_center = lesion_centroid
            else:
                crop_center = random_point if is_inside_body(image, random_point, hu_threshold) else lesion_centroid
        
        # Compute crop region (no offset needed, crop_center is already the target)
        region = compute_crop_region(crop_center, crop_size, image_size, spacing, (0, 0, 0), image)
        
        # Extract crops for both image and mask (with automatic padding if needed)
        image_crop = extract_crop(image, region, is_mask=False)
        mask_crop = extract_crop(mask, region, is_mask=True)
        
        crops.append((image_crop, mask_crop))
    
    return crops
