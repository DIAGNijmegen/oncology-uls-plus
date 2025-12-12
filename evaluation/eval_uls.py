import argparse
import csv
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import binary_erosion, binary_dilation
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
    m = re.search(r"_type-([^_]+)", name)
    return m.group(1) if m else "unknown"


def triad_key(name: str) -> str:
    return re.sub(r"_aug[12](?=\\.nii\\.gz$)", "", name)


def role(name: str) -> str:
    if re.search(r"_aug1(?=\\.nii\\.gz$)", name):
        return "aug1"
    if re.search(r"_aug2(?=\\.nii\\.gz$)", name):
        return "aug2"
    return "normal"


def stats(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, float)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def write_rows(rows: list[dict], out_csv: Path) -> None:
    cols = [
        "scope","lesion_type","n_cases","dsc_mean","dsc_std","biou_mean","biou_std",
        "n_triplets","agree_dsc_mean","agree_dsc_std","agree_biou_mean","agree_biou_std",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def _eval_one(pred_path: Path, label_path: Path) -> tuple[str, float, float] | None:
    if not label_path.exists():
        return None
    p = load_seg_bool(pred_path)
    g = load_seg_bool(label_path)
    return lesion_type(pred_path.name), dice(g, p), biou(g, p)


def _triad_one(key_name: str, normal_p: Path, aug1_p: Path, aug2_p: Path) -> tuple[str, float, float]:
    pn = load_seg_bool(normal_p)
    p1 = load_seg_bool(aug1_p)
    p2 = load_seg_bool(aug2_p)
    d = (dice(pn, p1) + dice(pn, p2) + dice(p1, p2)) / 3.0
    b = (biou(pn, p1) + biou(pn, p2) + biou(p1, p2)) / 3.0
    return lesion_type(key_name), float(d), float(b)


def _eval_one_tuple(args: tuple[Path, Path]) -> tuple[str, float, float] | None:
    return _eval_one(*args)


def _triad_one_tuple(args: tuple[str, Path, Path, Path]) -> tuple[str, float, float]:
    return _triad_one(*args)


def evaluate(dataset_root: Path, preds_dir: Path, out_csv: Path, workers: int = 1) -> None:
    labels_dir = dataset_root / "labelsTr"
    pred_files = sorted(preds_dir.glob("*.nii.gz"))
    eval_rec: list[tuple[str, float, float]] = []
    groups: dict[str, dict[str, Path]] = {}

    # Build jobs and role groups without I/O first
    eval_jobs: list[tuple[Path, Path]] = []
    for pf in pred_files:
        lf = labels_dir / pf.name
        eval_jobs.append((pf, lf))
        k = triad_key(pf.name); r = role(pf.name)
        groups.setdefault(k, {})[r] = pf

    # Evaluate per-file metrics in parallel
    if eval_jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in tqdm(ex.map(_eval_one_tuple, eval_jobs),
                            total=len(eval_jobs), desc="Evaluating predictions", unit="file"):
                if res is not None:
                    eval_rec.append(res)

    rows: list[dict] = []
    if eval_rec:
        by_t: dict[str, list[tuple[float, float]]] = {}
        for t, d, b in eval_rec:
            by_t.setdefault(t, []).append((d, b))
        all_d = [d for _, d, _ in eval_rec]; all_b = [b for _, _, b in eval_rec]
        md, sd, n = stats(all_d); mb, sb, _ = stats(all_b)
        rows.append({"scope":"evaluation","lesion_type":"ALL","n_cases":n,
                     "dsc_mean":md,"dsc_std":sd,"biou_mean":mb,"biou_std":sb})
        for t in sorted(by_t):
            dvals = [x[0] for x in by_t[t]]; bvals = [x[1] for x in by_t[t]]
            md, sd, n = stats(dvals); mb, sb, _ = stats(bvals)
            rows.append({"scope":"evaluation","lesion_type":t,"n_cases":n,
                        "dsc_mean":md,"dsc_std":sd,"biou_mean":mb,"biou_std":sb})

    triad: list[tuple[str, float, float]] = []
    triad_jobs: list[tuple[str, Path, Path, Path]] = []
    for k, rs in groups.items():
        if {"normal","aug1","aug2"}.issubset(rs):
            triad_jobs.append((k, rs["normal"], rs["aug1"], rs["aug2"]))

    if triad_jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for t_res in tqdm(ex.map(_triad_one_tuple, triad_jobs),
                               total=len(triad_jobs), desc="Computing agreement", unit="triplet"):
                triad.append(t_res)

    if triad:
        by_t2: dict[str, list[tuple[float, float]]] = {}
        for t, d, b in triad:
            by_t2.setdefault(t, []).append((d, b))
        all_d = [d for _, d, _ in triad]; all_b = [b for _, _, b in triad]
        md, sd, n = stats(all_d); mb, sb, _ = stats(all_b)
        rows.append({"scope":"agreement","lesion_type":"ALL","n_triplets":n,
                     "agree_dsc_mean":md,"agree_dsc_std":sd,
                     "agree_biou_mean":mb,"agree_biou_std":sb})
        for t in sorted(by_t2):
            dvals = [x[0] for x in by_t2[t]]; bvals = [x[1] for x in by_t2[t]]
            md, sd, n = stats(dvals); mb, sb, _ = stats(bvals)
            rows.append({"scope":"agreement","lesion_type":t,"n_triplets":n,
                        "agree_dsc_mean":md,"agree_dsc_std":sd,
                        "agree_biou_mean":mb,"agree_biou_std":sb})

    write_rows(rows, out_csv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128"))
    p.add_argument("--preds", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/preds"))
    p.add_argument("--out", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/uls_metrics.csv"))
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    evaluate(args.dataset_root, args.preds, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
