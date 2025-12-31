import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import binary_erosion, binary_dilation
import SimpleITK as sitk
from tqdm import tqdm


STRUCT = np.ones((3, 3, 3), dtype=bool)


def load_seg_bool(p: Path) -> np.ndarray:
    return nib.load(str(p)).get_fdata() > 0.5


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    sa = a.sum()
    sb = b.sum()
    if sa == 0 and sb == 0:
        return 1.0
    den = sa + sb
    return float(2.0 * inter / den) if den > 0 else 0.0


def bmask(m: np.ndarray) -> np.ndarray:
    return np.logical_xor(m, binary_erosion(m, structure=STRUCT, iterations=1))


def biou(a: np.ndarray, b: np.ndarray) -> float:
    ba = binary_dilation(bmask(a), structure=STRUCT, iterations=1)
    bb = binary_dilation(bmask(b), structure=STRUCT, iterations=1)
    uni = np.logical_or(ba, bb)
    u = uni.sum()
    if u == 0:
        return 1.0
    inter = np.logical_and(ba, bb).sum()
    return float(inter / u)


def lesion_type(name: str) -> str:
    # Extract type from _type-TYPE pattern, stopping before file extension
    m = re.search(r"_type-([^_.]+)", name)
    return m.group(1) if m else "unknown"


def triad_key(name: str) -> str:
    return re.sub(r"_aug[12](?=\.nii\.gz$)", "", name)


def role(name: str) -> str:
    if re.search(r"_aug1(?=\.nii\.gz$)", name):
        return "aug1"
    if re.search(r"_aug2(?=\.nii\.gz$)", name):
        return "aug2"
    return "normal"


def stats(vals: List[float]) -> Tuple[float, float, int]:
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, float)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def write_rows(rows: List[Dict], out_csv: Path) -> None:
    cols = [
        "scope","lesion_type","n_cases","n_unique_lesions","dsc_mean","dsc_std","biou_mean","biou_std",
        "n_triplets","n_unique_lesions_triplets","agree_dsc_mean","agree_dsc_std","agree_biou_mean","agree_biou_std",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def _eval_one(pred_path: Path, label_path: Path) -> Optional[Tuple[str, str, float, float]]:
    if not label_path.exists():
        return None
    p = load_seg_bool(pred_path)
    g = load_seg_bool(label_path)
    return pred_path.name, lesion_type(pred_path.name), dice(g, p), biou(g, p)


def compute_overlapping_region(img1_sitk: sitk.Image, img2_sitk: sitk.Image) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Compute overlapping region between two images in physical space.
    
    Returns:
        (start_idx_img1, start_idx_img2, overlap_size) where indices are in (x, y, z) voxel coordinates
        for each image, and overlap_size is the size of the overlapping region.
    """
    origin1 = np.array(img1_sitk.GetOrigin())
    origin2 = np.array(img2_sitk.GetOrigin())
    spacing1 = np.array(img1_sitk.GetSpacing())
    spacing2 = np.array(img2_sitk.GetSpacing())
    size1 = np.array(img1_sitk.GetSize())
    size2 = np.array(img2_sitk.GetSize())
    
    # Physical bounds: [min_physical, max_physical] for each image
    bounds1_min = origin1
    bounds1_max = origin1 + (size1 - 1) * spacing1
    bounds2_min = origin2
    bounds2_max = origin2 + (size2 - 1) * spacing2
    
    # Overlapping physical region
    overlap_min_physical = np.maximum(bounds1_min, bounds2_min)
    overlap_max_physical = np.minimum(bounds1_max, bounds2_max)
    
    # Check if there's any overlap
    if np.any(overlap_max_physical < overlap_min_physical):
        return (0, 0, 0), (0, 0, 0), (0, 0, 0)
    
    # Convert physical coordinates to voxel indices for each image
    # For image 1
    start1_physical = overlap_min_physical - origin1
    start1_voxel = np.round(start1_physical / spacing1).astype(int)
    start1_voxel = np.maximum(0, start1_voxel)
    
    end1_physical = overlap_max_physical - origin1
    end1_voxel = np.round(end1_physical / spacing1).astype(int) + 1
    end1_voxel = np.minimum(size1, end1_voxel)
    
    # For image 2
    start2_physical = overlap_min_physical - origin2
    start2_voxel = np.round(start2_physical / spacing2).astype(int)
    start2_voxel = np.maximum(0, start2_voxel)
    
    end2_physical = overlap_max_physical - origin2
    end2_voxel = np.round(end2_physical / spacing2).astype(int) + 1
    end2_voxel = np.minimum(size2, end2_voxel)
    
    # Compute overlap size (use minimum to ensure both fit)
    overlap_size = np.minimum(end1_voxel - start1_voxel, end2_voxel - start2_voxel)
    overlap_size = np.maximum(0, overlap_size)
    
    # Convert to tuples (SimpleITK uses x, y, z order)
    return tuple(start1_voxel), tuple(start2_voxel), tuple(overlap_size)


def extract_overlap_region(img_sitk: sitk.Image, start_idx: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
    """Extract a region from SimpleITK image and convert to numpy array in (x, y, z) order."""
    # Convert to list of unsigned ints for SimpleITK
    size_list = [int(s) for s in size]
    start_list = [int(s) for s in start_idx]
    
    # Check for zero-size dimensions
    if any(s <= 0 for s in size_list):
        # Return empty array with correct shape
        return np.zeros(size, dtype=np.float32)
    
    extract_filter = sitk.ExtractImageFilter()
    extract_filter.SetSize(size_list)
    extract_filter.SetIndex(start_list)
    extracted = extract_filter.Execute(img_sitk)
    arr = sitk.GetArrayFromImage(extracted)  # (z, y, x)
    return np.transpose(arr, (2, 1, 0))  # Transpose to (x, y, z)


def compute_three_way_overlap(img1_sitk: sitk.Image, img2_sitk: sitk.Image, img3_sitk: sitk.Image) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Compute common overlapping physical region between three images.
    
    Returns:
        (start_idx_img1, start_idx_img2, start_idx_img3, overlap_size) where indices
        are in (x, y, z) voxel coordinates for each image.
    """
    def get_bounds(img_sitk):
        origin = np.array(img_sitk.GetOrigin())
        spacing = np.array(img_sitk.GetSpacing())
        size = np.array(img_sitk.GetSize())
        return origin, origin + (size - 1) * spacing
    
    origin1, max1 = get_bounds(img1_sitk)
    origin2, max2 = get_bounds(img2_sitk)
    origin3, max3 = get_bounds(img3_sitk)
    
    # Common overlapping physical region
    overlap_min_physical = np.maximum.reduce([origin1, origin2, origin3])
    overlap_max_physical = np.minimum.reduce([max1, max2, max3])
    
    # Check if there's any overlap
    if np.any(overlap_max_physical < overlap_min_physical):
        return (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
    
    # Convert to voxel indices for each image
    def physical_to_voxel(origin, spacing, size, phys_min, phys_max):
        start_phys = phys_min - origin
        start_vox = np.round(start_phys / spacing).astype(int)
        start_vox = np.maximum(0, start_vox)
        
        end_phys = phys_max - origin
        end_vox = np.round(end_phys / spacing).astype(int) + 1
        end_vox = np.minimum(size, end_vox)
        
        overlap_size = end_vox - start_vox
        overlap_size = np.maximum(0, overlap_size)
        return tuple(start_vox), tuple(overlap_size)
    
    spacing1 = np.array(img1_sitk.GetSpacing())
    spacing2 = np.array(img2_sitk.GetSpacing())
    spacing3 = np.array(img3_sitk.GetSpacing())
    size1_arr = np.array(img1_sitk.GetSize())
    size2_arr = np.array(img2_sitk.GetSize())
    size3_arr = np.array(img3_sitk.GetSize())
    
    start1, size1 = physical_to_voxel(origin1, spacing1, size1_arr, overlap_min_physical, overlap_max_physical)
    start2, size2 = physical_to_voxel(origin2, spacing2, size2_arr, overlap_min_physical, overlap_max_physical)
    start3, size3 = physical_to_voxel(origin3, spacing3, size3_arr, overlap_min_physical, overlap_max_physical)
    
    # Use minimum size to ensure all three regions match
    size_min = tuple(min(s1, s2, s3) for s1, s2, s3 in zip(size1, size2, size3))
    
    return start1, start2, start3, size_min


def _triad_one(key_name: str, normal_p: Path, aug1_p: Path, aug2_p: Path) -> Tuple[str, float, float]:
    """
    Compute agreement between three predictions without resampling.
    
    Compares predictions in their common overlapping physical region to avoid
    interpolation artifacts. This ensures scientific rigor by comparing the
    same physical space across all three predictions.
    """
    # Load all three predictions with SimpleITK to preserve physical space
    normal_sitk = sitk.ReadImage(str(normal_p))
    aug1_sitk = sitk.ReadImage(str(aug1_p))
    aug2_sitk = sitk.ReadImage(str(aug2_p))
    
    # Find common overlapping region of all three
    start_n, start_a1, start_a2, overlap_size = compute_three_way_overlap(normal_sitk, aug1_sitk, aug2_sitk)
    
    # Check if overlap is too small (less than 10% of smallest dimension)
    min_dim = min(overlap_size) if overlap_size != (0, 0, 0) else 0
    min_original_size = min(normal_sitk.GetSize())
    if min_dim < min_original_size * 0.1:
        # Fallback: if overlap is too small, use pairwise comparisons
        # This shouldn't happen for properly cropped data, but handles edge cases
        start_n1, start_a1_1, size_n1a1 = compute_overlapping_region(normal_sitk, aug1_sitk)
        start_n2, start_a2_1, size_n2a2 = compute_overlapping_region(normal_sitk, aug2_sitk)
        start_a1a2_1, start_a1a2_2, size_a1a2 = compute_overlapping_region(aug1_sitk, aug2_sitk)
        
        pn_a1 = extract_overlap_region(normal_sitk, start_n1, size_n1a1) > 0.5
        pa1_n = extract_overlap_region(aug1_sitk, start_a1_1, size_n1a1) > 0.5
        pn_a2 = extract_overlap_region(normal_sitk, start_n2, size_n2a2) > 0.5
        pa2_n = extract_overlap_region(aug2_sitk, start_a2_1, size_n2a2) > 0.5
        pa1_a2_1 = extract_overlap_region(aug1_sitk, start_a1a2_1, size_a1a2) > 0.5
        pa1_a2_2 = extract_overlap_region(aug2_sitk, start_a1a2_2, size_a1a2) > 0.5
        
        d = (dice(pn_a1, pa1_n) + dice(pn_a2, pa2_n) + dice(pa1_a2_1, pa1_a2_2)) / 3.0
        b = (biou(pn_a1, pa1_n) + biou(pn_a2, pa2_n) + biou(pa1_a2_1, pa1_a2_2)) / 3.0
    else:
        # Extract common overlapping region from each prediction
        pn = extract_overlap_region(normal_sitk, start_n, overlap_size) > 0.5
        p1 = extract_overlap_region(aug1_sitk, start_a1, overlap_size) > 0.5
        p2 = extract_overlap_region(aug2_sitk, start_a2, overlap_size) > 0.5
        
        # Compute pairwise Dice and BIoU on the same overlapping region
        d = (dice(pn, p1) + dice(pn, p2) + dice(p1, p2)) / 3.0
        b = (biou(pn, p1) + biou(pn, p2) + biou(p1, p2)) / 3.0
    
    return lesion_type(key_name), float(d), float(b)


def _eval_one_tuple(args: Tuple[Path, Path]) -> Optional[Tuple[str, float, float]]:
    return _eval_one(*args)


def _triad_one_tuple(args: Tuple[str, Path, Path, Path]) -> Tuple[str, float, float]:
    return _triad_one(*args)


def write_per_sample_csv(sample_data: List[Tuple[str, float]], out_csv: Path) -> None:
    """Write per-sample Dice scores to CSV for statistical analysis."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "dice"])
        w.writeheader()
        for sample_name, dice_score in sorted(sample_data):
            w.writerow({"sample": sample_name, "dice": dice_score})


def write_per_sample_agreement_csv(agreement_data: List[Tuple[str, float, float]], out_csv: Path) -> None:
    """Write per-sample agreement scores to CSV for statistical analysis."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "agreement_dice", "agreement_biou"])
        w.writeheader()
        for sample_name, dice_score, biou_score in sorted(agreement_data):
            w.writerow({"sample": sample_name, "agreement_dice": dice_score, "agreement_biou": biou_score})


def evaluate(dataset_root: Path, preds_dir: Path, out_csv: Path, workers: int = 1, per_sample_csv: Optional[Path] = None, per_sample_agreement_csv: Optional[Path] = None, labels_dir: Optional[Path] = None) -> None:
    if labels_dir is None:
        labels_dir = dataset_root / "labelsTr"
    else:
        labels_dir = Path(labels_dir)
    pred_files = sorted(preds_dir.glob("*.nii.gz"))
    eval_rec: List[Tuple[str, str, float, float]] = []
    per_sample_data: List[Tuple[str, float]] = []
    per_sample_agreement: List[Tuple[str, float, float]] = []
    groups: Dict[str, Dict[str, Path]] = {}

    # Build jobs and role groups without I/O first
    eval_jobs: List[Tuple[Path, Path]] = []
    for pf in pred_files:
        # Transform prediction filename to match label filename
        # Predictions: "*_128_updated_..." -> Labels: "*_256_updated_..."
        label_name = pf.name.replace("_128_updated_", "_256_updated_")
        lf = labels_dir / label_name
        eval_jobs.append((pf, lf))
        k = triad_key(pf.name); r = role(pf.name)
        groups.setdefault(k, {})[r] = pf

    # Evaluate per-file metrics in parallel
    if eval_jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in tqdm(ex.map(_eval_one_tuple, eval_jobs),
                            total=len(eval_jobs), desc="Evaluating predictions", unit="file"):
                if res is not None:
                    sample_name, lesion_t, dice_score, biou_score = res
                    # Only include non-augmented samples for mean dice calculation
                    # But still evaluate aug1/aug2 for agreement calculations
                    sample_role = role(sample_name)
                    if sample_role == "normal":
                        eval_rec.append((lesion_t, dice_score, biou_score))
                        per_sample_data.append((sample_name, dice_score))

    # Write per-sample Dice CSV if requested (only non-augmented samples)
    if per_sample_csv and per_sample_data:
        write_per_sample_csv(per_sample_data, per_sample_csv)
    
    rows: List[Dict] = []
    if eval_rec:
        by_t: Dict[str, List[Tuple[float, float]]] = {}
        by_t_lesions: Dict[str, set] = {}  # Track unique lesions per type (only non-augmented)
        
        # Count unique lesions per type from non-augmented samples only
        for k, rs in groups.items():
            # Only count if "normal" (non-augmented) exists
            if "normal" in rs:
                sample_file = rs["normal"]
                t = lesion_type(sample_file.name)
                by_t_lesions.setdefault(t, set()).add(k)
        
        for t, d, b in eval_rec:
            by_t.setdefault(t, []).append((d, b))
        all_d = [d for _, d, _ in eval_rec]; all_b = [b for _, _, b in eval_rec]
        md, sd, n = stats(all_d); mb, sb, _ = stats(all_b)
        # Count unique lesions from non-augmented samples only
        all_unique_lesions = sum(1 for rs in groups.values() if "normal" in rs)
        rows.append({"scope":"evaluation","lesion_type":"ALL","n_cases":n,"n_unique_lesions":all_unique_lesions,
                     "dsc_mean":md,"dsc_std":sd,"biou_mean":mb,"biou_std":sb})
        for t in sorted(by_t):
            dvals = [x[0] for x in by_t[t]]; bvals = [x[1] for x in by_t[t]]
            md, sd, n = stats(dvals); mb, sb, _ = stats(bvals)
            n_unique = len(by_t_lesions.get(t, set()))
            rows.append({"scope":"evaluation","lesion_type":t,"n_cases":n,"n_unique_lesions":n_unique,
                        "dsc_mean":md,"dsc_std":sd,"biou_mean":mb,"biou_std":sb})

    triad: List[Tuple[str, float, float]] = []
    per_sample_agreement: List[Tuple[str, float, float]] = []
    triad_jobs: List[Tuple[str, Path, Path, Path]] = []
    for k, rs in groups.items():
        if {"normal","aug1","aug2"}.issubset(rs):
            triad_jobs.append((k, rs["normal"], rs["aug1"], rs["aug2"]))

    if triad_jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(_triad_one_tuple, triad_jobs),
                               total=len(triad_jobs), desc="Computing agreement", unit="triplet"))
            for (k, normal_p, _, _), t_res in zip(triad_jobs, results):
                lesion_t, dice_score, biou_score = t_res
                triad.append(t_res)
                # Get the base sample name from the normal file (without aug1/aug2 suffix)
                base_name = triad_key(normal_p.name)
                per_sample_agreement.append((base_name, dice_score, biou_score))

    if triad:
        by_t2: Dict[str, List[Tuple[float, float]]] = {}
        by_t2_lesions: Dict[str, set] = {}  # Track unique lesions per type for triplets
        for k, _, _, _ in triad_jobs:
            t = lesion_type(k)
            by_t2_lesions.setdefault(t, set()).add(k)
        
        for t, d, b in triad:
            by_t2.setdefault(t, []).append((d, b))
        all_d = [d for _, d, _ in triad]; all_b = [b for _, _, b in triad]
        md, sd, n = stats(all_d); mb, sb, _ = stats(all_b)
        all_unique_triplets = len(triad_jobs)
        rows.append({"scope":"agreement","lesion_type":"ALL","n_triplets":n,"n_unique_lesions_triplets":all_unique_triplets,
                     "agree_dsc_mean":md,"agree_dsc_std":sd,
                     "agree_biou_mean":mb,"agree_biou_std":sb})
        for t in sorted(by_t2):
            dvals = [x[0] for x in by_t2[t]]; bvals = [x[1] for x in by_t2[t]]
            md, sd, n = stats(dvals); mb, sb, _ = stats(bvals)
            n_unique = len(by_t2_lesions.get(t, set()))
            rows.append({"scope":"agreement","lesion_type":t,"n_triplets":n,"n_unique_lesions_triplets":n_unique,
                        "agree_dsc_mean":md,"agree_dsc_std":sd,
                        "agree_biou_mean":mb,"agree_biou_std":sb})
    
    # Write per-sample agreement CSV if requested
    if per_sample_agreement_csv and per_sample_agreement:
        write_per_sample_agreement_csv(per_sample_agreement, per_sample_agreement_csv)

    write_rows(rows, out_csv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128"))
    p.add_argument("--preds", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/preds"))
    p.add_argument("--labels", type=Path, default=None, help="Optional: custom labels directory (default: dataset-root/labelsTr)")
    p.add_argument("--out", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/uls_metrics.csv"))
    p.add_argument("--per-sample-csv", type=Path, default=None, help="Optional: path to write per-sample Dice scores CSV")
    p.add_argument("--per-sample-agreement-csv", type=Path, default=None, help="Optional: path to write per-sample agreement scores CSV")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    evaluate(args.dataset_root, args.preds, args.out, workers=args.workers, 
             per_sample_csv=args.per_sample_csv, per_sample_agreement_csv=args.per_sample_agreement_csv,
             labels_dir=args.labels)


if __name__ == "__main__":
    main()