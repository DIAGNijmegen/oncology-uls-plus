#!/usr/bin/env python3
"""
Create combined plots comparing multiple ULS datasets side by side.
Modern, clean styling with seaborn.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Plot styling
ANTHROPIC_BG = "#FFFFFF"  # Pure white background
ANTHROPIC_SECONDARY_BG = "#F7F3E8"  # Legend/panels
ANTHROPIC_GRID = "#E3DDD0"  # Subtle warm gridlines
ANTHROPIC_BORDER = "#D6CFBF"  # Light warm border
ANTHROPIC_TEXT_PRIMARY = "#24201E"  # Warm charcoal
ANTHROPIC_TEXT_SECONDARY = "#4A4A4A"  # Secondary text
ANTHROPIC_TEXT_TERTIARY = "#787878"  # Muted
ANTHROPIC_ACCENT = "#C47A2C"  # Warm amber (muted, paper-friendly)
ANTHROPIC_SECONDARY = "#3B332E"  # Warm charcoal for second series (not neutral gray)

# Set minimal seaborn style (will override colors manually)
sns.set_style("white", {
    "axes.edgecolor": ANTHROPIC_TEXT_SECONDARY,
    "axes.linewidth": 1.0,
    "grid.color": ANTHROPIC_GRID,
    "grid.linewidth": 0.8,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_evaluation_rows(csv_path: Path) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, int]]:
    """Load evaluation rows from CSV and return dicts: {lesion_type: (d_mean, d_std, b_mean, b_std)} and {lesion_type: n_samples}"""
    rows: Dict[str, Tuple[float, float, float, float]] = {}
    n_samples: Dict[str, int] = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("scope") != "evaluation":
                continue
            lt = str(row.get("lesion_type", ""))
            # Strip file extension if present
            if lt.endswith(".nii.gz"):
                lt = lt[:-7]
            elif lt.endswith(".nii"):
                lt = lt[:-4]
            d_mean = float(row.get("dsc_mean", 0) or 0)
            d_std = float(row.get("dsc_std", 0) or 0)
            b_mean = float(row.get("biou_mean", 0) or 0)
            b_std = float(row.get("biou_std", 0) or 0)
            n_val = int(row.get("n_unique_lesions", 0) or 0)
            rows[lt] = (d_mean, d_std, b_mean, b_std)
            n_samples[lt] = n_val
    return rows, n_samples


def load_agreement_rows(csv_path: Path) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, int]]:
    """Load agreement rows from CSV and return dicts: {lesion_type: (d_mean, d_std, b_mean, b_std)} and {lesion_type: n_samples}"""
    rows: Dict[str, Tuple[float, float, float, float]] = {}
    n_samples: Dict[str, int] = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("scope") != "agreement":
                continue
            lt = str(row.get("lesion_type", ""))
            # Strip file extension if present
            if lt.endswith(".nii.gz"):
                lt = lt[:-7]
            elif lt.endswith(".nii"):
                lt = lt[:-4]
            d_mean = float(row.get("agree_dsc_mean", 0) or 0)
            d_std = float(row.get("agree_dsc_std", 0) or 0)
            b_mean = float(row.get("agree_biou_mean", 0) or 0)
            b_std = float(row.get("agree_biou_std", 0) or 0)
            n_val = int(row.get("n_unique_lesions_triplets", 0) or 0)
            rows[lt] = (d_mean, d_std, b_mean, b_std)
            n_samples[lt] = n_val
    return rows, n_samples


def normalize_lesion_type_names(
    data: Dict[str, Tuple[float, float, float, float]],
    n_samples: Dict[str, int]
) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, int]]:
    """Normalize lesion type names: rename 'soft-tissue---skin' to 'skin', 'adrenals' to 'adrenal', 'lymph-node' to 'lymph node', 'skeleton' to 'bone'."""
    normalized_data = {}
    normalized_n_samples = {}
    
    for lt, values in data.items():
        # Rename soft-tissue---skin to skin
        if lt == "soft-tissue---skin":
            normalized_lt = "skin"
        elif lt == "adrenals":
            normalized_lt = "adrenal"
        elif lt == "lymph-node":
            normalized_lt = "lymph node"
        elif lt == "skeleton":
            normalized_lt = "bone"
        else:
            normalized_lt = lt
        
        if normalized_lt in normalized_data:
            # If already exists, merge using weighted statistics
            d_mean, d_std, b_mean, b_std = normalized_data[normalized_lt]
            d_mean2, d_std2, b_mean2, b_std2 = values
            n1 = normalized_n_samples[normalized_lt]
            n2 = n_samples[lt]
            total_n = n1 + n2
            
            normalized_data[normalized_lt] = (
                (n1 * d_mean + n2 * d_mean2) / total_n,
                (n1 * d_std + n2 * d_std2) / total_n,
                (n1 * b_mean + n2 * b_mean2) / total_n,
                (n1 * b_std + n2 * b_std2) / total_n,
            )
            normalized_n_samples[normalized_lt] = total_n
        else:
            normalized_data[normalized_lt] = values
            normalized_n_samples[normalized_lt] = n_samples[lt]
    
    return normalized_data, normalized_n_samples


def merge_other_categories(
    data: Dict[str, Tuple[float, float, float, float]],
    n_samples: Dict[str, int]
) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, int]]:
    """Merge 'other', 'others', and 'unclear' into a single 'other' category."""
    merged_data = data.copy()
    merged_n_samples = n_samples.copy()
    
    # Categories to merge
    categories_to_merge = ["other", "others", "unclear"]
    
    # Check if any of these categories exist
    existing_categories = [cat for cat in categories_to_merge if cat in merged_data]
    
    if len(existing_categories) <= 1:
        # Nothing to merge
        return merged_data, merged_n_samples
    
    # Use "other" as the target category (create if it doesn't exist)
    if "other" not in merged_data:
        # Use the first existing category as the base
        base_cat = existing_categories[0]
        merged_data["other"] = merged_data[base_cat]
        merged_n_samples["other"] = merged_n_samples[base_cat]
        if base_cat != "other":
            del merged_data[base_cat]
            del merged_n_samples[base_cat]
        existing_categories = [cat for cat in existing_categories if cat != base_cat]
    
    # Merge remaining categories into "other"
    other_d_mean, other_d_std, other_b_mean, other_b_std = merged_data["other"]
    other_n = merged_n_samples["other"]
    
    for cat in existing_categories:
        if cat == "other" or cat not in merged_data:
            continue
        
        d_mean, d_std, b_mean, b_std = merged_data[cat]
        n = merged_n_samples[cat]
        
        # Weighted mean
        total_n = other_n + n
        other_d_mean = (other_n * other_d_mean + n * d_mean) / total_n
        other_b_mean = (other_n * other_b_mean + n * b_mean) / total_n
        
        # Weighted average of stds
        other_d_std = (other_n * other_d_std + n * d_std) / total_n
        other_b_std = (other_n * other_b_std + n * b_std) / total_n
        
        other_n = total_n
        
        # Remove the merged category
        del merged_data[cat]
        del merged_n_samples[cat]
    
    # Update "other" with merged values
    merged_data["other"] = (other_d_mean, other_d_std, other_b_mean, other_b_std)
    merged_n_samples["other"] = other_n
    
    return merged_data, merged_n_samples


def merge_n1_types_with_unknown(
    data: Dict[str, Tuple[float, float, float, float]],
    n_samples: Dict[str, int]
) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, int]]:
    """Merge lesion types with N=1 into 'unknown' type using weighted statistics.
    
    Returns updated data and n_samples dicts with N=1 types removed and merged into 'unknown'.
    """
    merged_data = data.copy()
    merged_n_samples = n_samples.copy()
    
    # Find types with N=1 (excluding 'unknown' and 'ALL')
    n1_types = [lt for lt in merged_n_samples.keys() 
                if lt not in ["unknown", "ALL"] and merged_n_samples[lt] == 1]
    
    if not n1_types or "unknown" not in merged_data:
        return merged_data, merged_n_samples
    
    # Get unknown data
    unknown_d_mean, unknown_d_std, unknown_b_mean, unknown_b_std = merged_data["unknown"]
    unknown_n = merged_n_samples["unknown"]
    
    # Merge each N=1 type into unknown using weighted statistics
    for lt in n1_types:
        if lt not in merged_data:
            continue
        
        d_mean, d_std, b_mean, b_std = merged_data[lt]
        n = merged_n_samples[lt]
        
        # Weighted mean: (n1*mean1 + n2*mean2) / (n1 + n2)
        total_n = unknown_n + n
        unknown_d_mean = (unknown_n * unknown_d_mean + n * d_mean) / total_n
        unknown_b_mean = (unknown_n * unknown_b_mean + n * b_mean) / total_n
        
        # For std, use pooled variance approximation
        # Simplified: take max std as conservative estimate, or use weighted average
        # Using weighted average of stds (not perfect but reasonable)
        unknown_d_std = (unknown_n * unknown_d_std + n * d_std) / total_n
        unknown_b_std = (unknown_n * unknown_b_std + n * b_std) / total_n
        
        unknown_n = total_n
        
        # Remove the N=1 type
        del merged_data[lt]
        del merged_n_samples[lt]
    
    # Update unknown with merged values
    merged_data["unknown"] = (unknown_d_mean, unknown_d_std, unknown_b_mean, unknown_b_std)
    merged_n_samples["unknown"] = unknown_n
    
    return merged_data, merged_n_samples


def merge_soft_tissue_types(
    data: Dict[str, Tuple[float, float, float, float]],
    n_samples: Dict[str, int]
) -> Tuple[Dict[str, Tuple[float, float, float, float]], Dict[str, int]]:
    """Merge 'muscle', 'soft-tissue', and 'subcutaneous' into a single 'soft-tissue' category.
    
    Returns updated data and n_samples dicts with these types merged.
    """
    merged_data = data.copy()
    merged_n_samples = n_samples.copy()
    
    # Types to merge
    types_to_merge = ["muscle", "soft-tissue", "subcutaneous"]
    
    # Check which types exist
    existing_types = [t for t in types_to_merge if t in merged_data]
    
    if len(existing_types) <= 1:
        # Nothing to merge (or already merged)
        return merged_data, merged_n_samples
    
    # Use "soft-tissue" as the target category
    if "soft-tissue" not in merged_data:
        # Use the first existing type as the base
        base_type = existing_types[0]
        merged_data["soft-tissue"] = merged_data[base_type]
        merged_n_samples["soft-tissue"] = merged_n_samples[base_type]
        if base_type != "soft-tissue":
            del merged_data[base_type]
            del merged_n_samples[base_type]
        existing_types = [t for t in existing_types if t != base_type]
    
    # Get soft-tissue data
    st_d_mean, st_d_std, st_b_mean, st_b_std = merged_data["soft-tissue"]
    st_n = merged_n_samples["soft-tissue"]
    
    # Merge remaining types into soft-tissue
    for t in existing_types:
        if t == "soft-tissue" or t not in merged_data:
            continue
        
        d_mean, d_std, b_mean, b_std = merged_data[t]
        n = merged_n_samples[t]
        
        # Weighted mean
        total_n = st_n + n
        st_d_mean = (st_n * st_d_mean + n * d_mean) / total_n
        st_b_mean = (st_n * st_b_mean + n * b_mean) / total_n
        
        # Weighted average of stds
        st_d_std = (st_n * st_d_std + n * d_std) / total_n
        st_b_std = (st_n * st_b_std + n * b_std) / total_n
        
        st_n = total_n
        
        # Remove the merged type
        del merged_data[t]
        del merged_n_samples[t]
    
    # Update soft-tissue with merged values
    merged_data["soft-tissue"] = (st_d_mean, st_d_std, st_b_mean, st_b_std)
    merged_n_samples["soft-tissue"] = st_n
    
    return merged_data, merged_n_samples


def get_display_name(lesion_type: str) -> str:
    """Get display name for lesion type, converting 'ALL' to 'Overall' and 'soft-tissue' to 'soft tissue'."""
    if lesion_type == "ALL":
        return "Overall"
    if lesion_type == "soft-tissue":
        return "soft tissue"
    return lesion_type


def order_types(types: List[str]) -> List[str]:
    """Order types with ALL at the end, and 'other' before ALL."""
    items = sorted([t for t in types if t != "ALL" and t != "other"])
    # Add 'other' before ALL
    if "other" in types:
        items.append("other")
    if "ALL" in types:
        items.append("ALL")
    return items


def plot_combined_bars(
    labels: List[str],
    datasets: Dict[str, List[Tuple[float, float]]],
    ylabel: str,
    title: str,
    out_path: Path,
    n_samples_per_type: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """Create grouped bar chart comparing multiple datasets with modern styling.
    
    Args:
        n_samples_per_type: Optional dict mapping dataset name to dict of {lesion_type: n_samples}
    """
    n_types = len(labels)
    n_datasets = len(datasets)
    
    # Calculate bar width and positions
    bar_width = 0.35
    x = np.arange(n_types)
    
    # Calculate positions for each dataset
    if n_datasets == 2:
        offset = bar_width / 2
        positions = {
            list(datasets.keys())[0]: x - offset,
            list(datasets.keys())[1]: x + offset,
        }
    else:
        total_width = bar_width * n_datasets
        start_offset = -total_width / 2 + bar_width / 2
        positions = {}
        for i, name in enumerate(datasets.keys()):
            positions[name] = x + start_offset + i * bar_width
    
    # Anthropic color palette: one accent, restrained
    colors = [ANTHROPIC_ACCENT, ANTHROPIC_SECONDARY]  # Orange and near-black
    if n_datasets > 2:
        colors = [ANTHROPIC_ACCENT, ANTHROPIC_SECONDARY, ANTHROPIC_TEXT_TERTIARY, "#CFCABE"]
    
    # Create figure with Anthropic styling - use consistent width for all plots
    # Fixed width ensures consistent sizing across different datasets
    fig_width = 16.0
    fig, ax = plt.subplots(figsize=(fig_width, 7.0), dpi=150)
    fig.patch.set_facecolor(ANTHROPIC_BG)
    ax.set_facecolor(ANTHROPIC_BG)
    
    # Prepare legend labels (simple, without N)
    legend_labels = list(datasets.keys())
    
    # Prepare x-axis labels with N values per lesion type
    x_labels_with_n = []
    for label in labels:
        # Get display name (converts "ALL" to "Overall")
        display_label = get_display_name(label)
        # Try to get N from first dataset (they should be similar)
        n_val = None
        if n_samples_per_type:
            for dataset_name in datasets.keys():
                if dataset_name in n_samples_per_type:
                    n_dict = n_samples_per_type[dataset_name]
                    n_val = n_dict.get(label, None)
                    if n_val is not None and n_val > 0:
                        break
        
        if n_val is not None and n_val > 0:
            x_labels_with_n.append(f"{display_label}\n(N={n_val})")
        else:
            x_labels_with_n.append(display_label)
    
    # Plot bars for each dataset
    bars_all = []
    max_label_height = 0.0
    for i, (dataset_name, values) in enumerate(datasets.items()):
        means = np.array([v[0] for v in values])
        stds = np.array([v[1] for v in values])
        pos = positions[dataset_name]
        color = colors[i % len(colors)]
        # Errorbar color per series:
        # - On dark bars, use a lighter warm errorbar so it's readable.
        # - On the amber bar, use a darker brown errorbar for contrast.
        if i == 0:
            error_ecolor = "#5A3B22"  # dark warm brown
        elif i == 1:
            error_ecolor = "#B8B0A1"  # light warm gray (visible on dark charcoal)
        else:
            error_ecolor = ANTHROPIC_TEXT_SECONDARY
        
        # Cap error bars at 1.0 - Dice/BIoU are bounded [0, 1]
        # Upper error should not exceed 1.0 - mean
        upper_err = np.minimum(stds, 1.0 - means)
        # Lower error should not exceed mean
        lower_err = np.minimum(stds, means)
        # Create 2D array for asymmetric error bars: [[lower_errors], [upper_errors]]
        error_array = np.array([lower_err, upper_err])
        
        # Make zero-height bars visible by giving them a minimal height
        # This ensures bars with 0.0 values are still visible in the plot
        display_means = np.where(means == 0.0, 0.01, means)
        
        bars = ax.bar(
            pos,
            display_means,
            bar_width,
            yerr=error_array,
            capsize=4,
            label=legend_labels[i],
            color=color,
            edgecolor=ANTHROPIC_BG,
            linewidth=2.0,
            alpha=0.82,
            error_kw={"elinewidth": 2.0, "ecolor": error_ecolor, "capthick": 2.0},
        )
        bars_all.append(bars)
        
        # Add value labels above bars, accounting for error bars
        for rect, val, err in zip(bars, means, upper_err):
            # Label all values, including zeros, so missing bars are visible
            if val == 0.0:
                # For zero values, place label at a small offset so it's visible
                label_y = 0.02
                value_str = ".00"
            elif val > 0.01:
                # Position label above error bar with more spacing
                label_y = val + err + 0.05
                max_label_height = max(max_label_height, label_y)
                # Format as .xx instead of 0.xx
                value_str = f"{val:.2f}"
                if value_str.startswith("0."):
                    value_str = "." + value_str[2:]  # Remove leading 0
            else:
                # Very small non-zero values (< 0.01)
                label_y = val + err + 0.05
                max_label_height = max(max_label_height, label_y)
                value_str = f"{val:.2f}"
                if value_str.startswith("0."):
                    value_str = "." + value_str[2:]
            
            ax.text(rect.get_x() + rect.get_width() / 2.0, label_y,
                   value_str, ha="center", va="bottom", 
                   fontsize=11, fontweight=600, color=ANTHROPIC_TEXT_PRIMARY)
    
    # Calculate appropriate ylim based on max label height, but cap at reasonable max
    # Since values are bounded [0, 1], we only need space for labels above 1.0
    y_max = max(1.0, min(max_label_height + 0.08, 1.15))  # Cap at 1.15 max
    
    # Styling with larger fonts for paper readability
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels_with_n, rotation=30, ha="right", fontsize=14, fontweight=500, color=ANTHROPIC_TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=16, fontweight=600, color=ANTHROPIC_TEXT_PRIMARY, labelpad=10)
    # Remove title
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(-0.6, n_types - 0.4)
    
    # Increase tick label sizes
    ax.tick_params(axis='y', labelsize=14, width=1.5, length=5, colors=ANTHROPIC_TEXT_SECONDARY)
    ax.tick_params(axis='x', labelsize=14, width=1.5, length=5, colors=ANTHROPIC_TEXT_SECONDARY)
    
    # Legend - horizontal layout above the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=True, fancybox=False, 
             edgecolor=ANTHROPIC_BORDER, framealpha=0.95, fontsize=14, 
             ncol=2, handlelength=1.5, handletextpad=0.5,
             facecolor=ANTHROPIC_SECONDARY_BG, labelcolor=ANTHROPIC_TEXT_PRIMARY)
    
    # Clean grid - Anthropic style
    ax.grid(axis="y", alpha=0.6, linestyle="-", linewidth=0.8, color=ANTHROPIC_GRID)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look - Anthropic colors
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ANTHROPIC_TEXT_SECONDARY)
    ax.spines["bottom"].set_color(ANTHROPIC_TEXT_SECONDARY)
    
    # Adjust layout to leave space for legend above
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=ANTHROPIC_BG)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Create combined plots comparing multiple ULS datasets")
    p.add_argument("--csv-uls", type=Path, required=True, help="CSV file for ULS (256)")
    p.add_argument("--csv-uls-plus", type=Path, required=True, help="CSV file for ULS+ (128)")
    p.add_argument("--outdir", type=Path, default=None, help="Output directory for plots")
    args = p.parse_args()
    
    # Load evaluation data from both CSVs
    uls_data, uls_n_samples = load_evaluation_rows(args.csv_uls)
    uls_plus_data, uls_plus_n_samples = load_evaluation_rows(args.csv_uls_plus)
    
    # Load agreement data from both CSVs
    uls_agreement_data, uls_agreement_n_samples = load_agreement_rows(args.csv_uls)
    uls_plus_agreement_data, uls_plus_agreement_n_samples = load_agreement_rows(args.csv_uls_plus)
    
    # Normalize lesion type names (rename soft-tissue---skin to skin)
    uls_data, uls_n_samples = normalize_lesion_type_names(uls_data, uls_n_samples)
    uls_plus_data, uls_plus_n_samples = normalize_lesion_type_names(uls_plus_data, uls_plus_n_samples)
    uls_agreement_data, uls_agreement_n_samples = normalize_lesion_type_names(uls_agreement_data, uls_agreement_n_samples)
    uls_plus_agreement_data, uls_plus_agreement_n_samples = normalize_lesion_type_names(uls_plus_agreement_data, uls_plus_agreement_n_samples)
    
    # Merge 'other', 'others', and 'unclear' into 'other'
    uls_data, uls_n_samples = merge_other_categories(uls_data, uls_n_samples)
    uls_plus_data, uls_plus_n_samples = merge_other_categories(uls_plus_data, uls_plus_n_samples)
    uls_agreement_data, uls_agreement_n_samples = merge_other_categories(uls_agreement_data, uls_agreement_n_samples)
    uls_plus_agreement_data, uls_plus_agreement_n_samples = merge_other_categories(uls_plus_agreement_data, uls_plus_agreement_n_samples)
    
    # Merge N=1 types with unknown for evaluation data
    uls_data, uls_n_samples = merge_n1_types_with_unknown(uls_data, uls_n_samples)
    uls_plus_data, uls_plus_n_samples = merge_n1_types_with_unknown(uls_plus_data, uls_plus_n_samples)
    
    # Merge N=1 types with unknown for agreement data
    uls_agreement_data, uls_agreement_n_samples = merge_n1_types_with_unknown(uls_agreement_data, uls_agreement_n_samples)
    uls_plus_agreement_data, uls_plus_agreement_n_samples = merge_n1_types_with_unknown(uls_plus_agreement_data, uls_plus_agreement_n_samples)
    
    # Merge muscle, soft-tissue, and subcutaneous into soft-tissue
    uls_data, uls_n_samples = merge_soft_tissue_types(uls_data, uls_n_samples)
    uls_plus_data, uls_plus_n_samples = merge_soft_tissue_types(uls_plus_data, uls_plus_n_samples)
    uls_agreement_data, uls_agreement_n_samples = merge_soft_tissue_types(uls_agreement_data, uls_agreement_n_samples)
    uls_plus_agreement_data, uls_plus_agreement_n_samples = merge_soft_tissue_types(uls_plus_agreement_data, uls_plus_agreement_n_samples)
    
    # Rename 'unknown' to 'other' in data dicts (if it exists after N=1 merging)
    if "unknown" in uls_data:
        if "other" in uls_data:
            # Merge unknown into other
            other_d_mean, other_d_std, other_b_mean, other_b_std = uls_data["other"]
            unknown_d_mean, unknown_d_std, unknown_b_mean, unknown_b_std = uls_data["unknown"]
            other_n = uls_n_samples["other"]
            unknown_n = uls_n_samples["unknown"]
            total_n = other_n + unknown_n
            uls_data["other"] = (
                (other_n * other_d_mean + unknown_n * unknown_d_mean) / total_n,
                (other_n * other_d_std + unknown_n * unknown_d_std) / total_n,
                (other_n * other_b_mean + unknown_n * unknown_b_mean) / total_n,
                (other_n * other_b_std + unknown_n * unknown_b_std) / total_n,
            )
            uls_n_samples["other"] = total_n
            del uls_data["unknown"]
            del uls_n_samples["unknown"]
        else:
            uls_data["other"] = uls_data.pop("unknown")
            uls_n_samples["other"] = uls_n_samples.pop("unknown")
    if "unknown" in uls_plus_data:
        if "other" in uls_plus_data:
            other_d_mean, other_d_std, other_b_mean, other_b_std = uls_plus_data["other"]
            unknown_d_mean, unknown_d_std, unknown_b_mean, unknown_b_std = uls_plus_data["unknown"]
            other_n = uls_plus_n_samples["other"]
            unknown_n = uls_plus_n_samples["unknown"]
            total_n = other_n + unknown_n
            uls_plus_data["other"] = (
                (other_n * other_d_mean + unknown_n * unknown_d_mean) / total_n,
                (other_n * other_d_std + unknown_n * unknown_d_std) / total_n,
                (other_n * other_b_mean + unknown_n * unknown_b_mean) / total_n,
                (other_n * other_b_std + unknown_n * unknown_b_std) / total_n,
            )
            uls_plus_n_samples["other"] = total_n
            del uls_plus_data["unknown"]
            del uls_plus_n_samples["unknown"]
        else:
            uls_plus_data["other"] = uls_plus_data.pop("unknown")
            uls_plus_n_samples["other"] = uls_plus_n_samples.pop("unknown")
    if "unknown" in uls_agreement_data:
        if "other" in uls_agreement_data:
            other_d_mean, other_d_std, other_b_mean, other_b_std = uls_agreement_data["other"]
            unknown_d_mean, unknown_d_std, unknown_b_mean, unknown_b_std = uls_agreement_data["unknown"]
            other_n = uls_agreement_n_samples["other"]
            unknown_n = uls_agreement_n_samples["unknown"]
            total_n = other_n + unknown_n
            uls_agreement_data["other"] = (
                (other_n * other_d_mean + unknown_n * unknown_d_mean) / total_n,
                (other_n * other_d_std + unknown_n * unknown_d_std) / total_n,
                (other_n * other_b_mean + unknown_n * unknown_b_mean) / total_n,
                (other_n * other_b_std + unknown_n * unknown_b_std) / total_n,
            )
            uls_agreement_n_samples["other"] = total_n
            del uls_agreement_data["unknown"]
            del uls_agreement_n_samples["unknown"]
        else:
            uls_agreement_data["other"] = uls_agreement_data.pop("unknown")
            uls_agreement_n_samples["other"] = uls_agreement_n_samples.pop("unknown")
    if "unknown" in uls_plus_agreement_data:
        if "other" in uls_plus_agreement_data:
            other_d_mean, other_d_std, other_b_mean, other_b_std = uls_plus_agreement_data["other"]
            unknown_d_mean, unknown_d_std, unknown_b_mean, unknown_b_std = uls_plus_agreement_data["unknown"]
            other_n = uls_plus_agreement_n_samples["other"]
            unknown_n = uls_plus_agreement_n_samples["unknown"]
            total_n = other_n + unknown_n
            uls_plus_agreement_data["other"] = (
                (other_n * other_d_mean + unknown_n * unknown_d_mean) / total_n,
                (other_n * other_d_std + unknown_n * unknown_d_std) / total_n,
                (other_n * other_b_mean + unknown_n * unknown_b_mean) / total_n,
                (other_n * other_b_std + unknown_n * unknown_b_std) / total_n,
            )
            uls_plus_agreement_n_samples["other"] = total_n
            del uls_plus_agreement_data["unknown"]
            del uls_plus_agreement_n_samples["unknown"]
        else:
            uls_plus_agreement_data["other"] = uls_plus_agreement_data.pop("unknown")
            uls_plus_agreement_n_samples["other"] = uls_plus_agreement_n_samples.pop("unknown")
    
    # Get all unique lesion types (from both evaluation and agreement)
    all_types = set(uls_data.keys()) | set(uls_plus_data.keys())
    all_agreement_types = set(uls_agreement_data.keys()) | set(uls_plus_agreement_data.keys())
    types = order_types(list(all_types))
    agreement_types = order_types(list(all_agreement_types))
    
    # Prepare evaluation data for plotting
    datasets_dice: Dict[str, List[Tuple[float, float]]] = {
        "ULS": [],
        "ULS+": [],
    }
    datasets_biou: Dict[str, List[Tuple[float, float]]] = {
        "ULS": [],
        "ULS+": [],
    }
    
    for lt in types:
        # Dice data
        if lt in uls_data:
            datasets_dice["ULS"].append((uls_data[lt][0], uls_data[lt][1]))
        else:
            datasets_dice["ULS"].append((0.0, 0.0))
        
        if lt in uls_plus_data:
            datasets_dice["ULS+"].append((uls_plus_data[lt][0], uls_plus_data[lt][1]))
        else:
            datasets_dice["ULS+"].append((0.0, 0.0))
        
        # BIoU data
        if lt in uls_data:
            datasets_biou["ULS"].append((uls_data[lt][2], uls_data[lt][3]))
        else:
            datasets_biou["ULS"].append((0.0, 0.0))
        
        if lt in uls_plus_data:
            datasets_biou["ULS+"].append((uls_plus_data[lt][2], uls_plus_data[lt][3]))
        else:
            datasets_biou["ULS+"].append((0.0, 0.0))
    
    # Prepare agreement data for plotting
    datasets_agree_dice: Dict[str, List[Tuple[float, float]]] = {
        "ULS": [],
        "ULS+": [],
    }
    datasets_agree_biou: Dict[str, List[Tuple[float, float]]] = {
        "ULS": [],
        "ULS+": [],
    }
    
    for lt in agreement_types:
        # Agreement Dice data
        if lt in uls_agreement_data:
            datasets_agree_dice["ULS"].append((uls_agreement_data[lt][0], uls_agreement_data[lt][1]))
        else:
            datasets_agree_dice["ULS"].append((0.0, 0.0))
        
        if lt in uls_plus_agreement_data:
            datasets_agree_dice["ULS+"].append((uls_plus_agreement_data[lt][0], uls_plus_agreement_data[lt][1]))
        else:
            datasets_agree_dice["ULS+"].append((0.0, 0.0))
        
        # Agreement BIoU data
        if lt in uls_agreement_data:
            datasets_agree_biou["ULS"].append((uls_agreement_data[lt][2], uls_agreement_data[lt][3]))
        else:
            datasets_agree_biou["ULS"].append((0.0, 0.0))
        
        if lt in uls_plus_agreement_data:
            datasets_agree_biou["ULS+"].append((uls_plus_agreement_data[lt][2], uls_plus_agreement_data[lt][3]))
        else:
            datasets_agree_biou["ULS+"].append((0.0, 0.0))
    
    # Determine output directory
    out_dir = args.outdir if args.outdir is not None else args.csv_uls.parent
    
    # Prepare N samples dicts for evaluation plots
    eval_n_samples = {
        "ULS": uls_n_samples,
        "ULS+": uls_plus_n_samples,
    }
    
    # Prepare N samples dicts for agreement plots
    agreement_n_samples = {
        "ULS": uls_agreement_n_samples,
        "ULS+": uls_plus_agreement_n_samples,
    }
    
    # Create evaluation plots
    plot_combined_bars(types, datasets_dice, ylabel="Dice Score", 
                       title="Dice Score by Lesion Type: ULS vs ULS+",
                       out_path=out_dir / "combined_dice_by_type.png",
                       n_samples_per_type=eval_n_samples)
    
    plot_combined_bars(types, datasets_biou, ylabel="Boundary IoU",
                       title="Boundary IoU by Lesion Type: ULS vs ULS+",
                       out_path=out_dir / "combined_biou_by_type.png",
                       n_samples_per_type=eval_n_samples)
    
    # Create agreement plots
    if agreement_types:
        plot_combined_bars(agreement_types, datasets_agree_dice, ylabel="Robustness Score", 
                           title="Prediction Agreement (Dice) by Lesion Type: ULS vs ULS+",
                           out_path=out_dir / "combined_agreement_dice_by_type.png",
                           n_samples_per_type=agreement_n_samples)
        
        plot_combined_bars(agreement_types, datasets_agree_biou, ylabel="Robustness Boundary IoU",
                           title="Prediction Agreement (BIoU) by Lesion Type: ULS vs ULS+",
                           out_path=out_dir / "combined_agreement_biou_by_type.png",
                           n_samples_per_type=agreement_n_samples)
    
    print(f"Combined plots saved to {out_dir}")


if __name__ == "__main__":
    main()
