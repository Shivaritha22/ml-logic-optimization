"""
Reads the MTCNN trace log and produces 3 plots:
1. Bar chart - average rejection rate per decision point
2. Funnel chart - average candidate count at each stage
3. Box plot - distribution of rejection rates across all images
Usage: python scripts/analyze_plot.py logs/mtcnn_trace_XXXXXX.json
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# ---------------------------------------------------------------
# Load trace log
# ---------------------------------------------------------------

def load_traces(log_path):
    """
    Load trace entries from a formatted JSON trace (DecisionTracer.save)
    or legacy JSONL (one JSON object per line).
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.strip()
    if not text:
        print(f"Loaded 0 entries from {log_path}")
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and 'entries' in payload:
        entries = payload['entries']
        n = payload.get('entry_count', len(entries))
        print(f"Loaded {len(entries)} entries from {log_path} (JSON trace, entry_count={n})")
        return entries

    entries = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    print(f"Loaded {len(entries)} entries from {log_path} (JSONL)")
    return entries


def group_by_image_and_stage(entries):
    """
    For each image, collect ONE entry per stage.
    pnet_threshold appears multiple times per image (once per scale).
    We aggregate those by summing input/output counts.
    Returns: dict of {image_id: {stage: {input, output, rejected}}}
    """
    # First pass: group by image_id and stage
    raw = defaultdict(lambda: defaultdict(list))
    for e in entries:
        raw[e['image_id']][e['stage']].append(e)

    # Second pass: aggregate pnet_threshold and pnet_nms_scale (multi-scale)
    aggregated = {}
    for image_id, stages in raw.items():
        aggregated[image_id] = {}
        for stage, stage_entries in stages.items():
            if len(stage_entries) == 1:
                aggregated[image_id][stage] = {
                    'input_count':    stage_entries[0]['input_count'],
                    'output_count':   stage_entries[0]['output_count'],
                    'rejected_count': stage_entries[0]['rejected_count'],
                    'value':          stage_entries[0]['value'],
                }
            else:
                # Multiple entries (pnet_threshold, pnet_nms_scale) - sum them
                aggregated[image_id][stage] = {
                    'input_count':    sum(e['input_count'] for e in stage_entries),
                    'output_count':   sum(e['output_count'] for e in stage_entries),
                    'rejected_count': sum(e['rejected_count'] for e in stage_entries),
                    'value':          stage_entries[0]['value'],
                }
    return aggregated


# ---------------------------------------------------------------
# Plot 1: Bar chart - average rejection rate per decision point
# ---------------------------------------------------------------

def plot_rejection_rates(aggregated, output_dir):
    """
    Bar chart showing average rejection rate (%) per decision point.
    Rejection rate = rejected_count / input_count * 100
    """

    stage_order = [
        'pnet_threshold',
        'pnet_nms_scale',
        'pnet_nms_cross',
        'rnet_threshold',
        'rnet_nms',
        'onet_threshold',
        'onet_nms',
    ]

    stage_labels = [
        'P-Net\nThreshold',
        'P-Net NMS\n(per scale)',
        'P-Net NMS\n(cross scale)',
        'R-Net\nThreshold',
        'R-Net\nNMS',
        'O-Net\nThreshold',
        'O-Net\nNMS',
    ]

    rejection_rates = []
    for stage in stage_order:
        rates = []
        for image_data in aggregated.values():
            if stage in image_data:
                entry = image_data[stage]
                if entry['input_count'] > 0:
                    rate = entry['rejected_count'] / entry['input_count'] * 100
                    rates.append(rate)
        avg_rate = np.mean(rates) if rates else 0
        rejection_rates.append(avg_rate)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#2196F3'] * 3 + ['#4CAF50'] * 2 + ['#FF9800'] * 2
    bars = ax.bar(stage_labels, rejection_rates, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on top of each bar
    for bar, rate in zip(bars, rejection_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{rate:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color='#2196F3', label='P-Net decisions'),
        mpatches.Patch(color='#4CAF50', label='R-Net decisions'),
        mpatches.Patch(color='#FF9800', label='O-Net decisions'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    ax.set_title('Average Rejection Rate per Decision Point\n(across 1000 CelebA images)', fontsize=14)
    ax.set_ylabel('Average Rejection Rate (%)', fontsize=12)
    ax.set_xlabel('Decision Point', fontsize=12)
    ax.set_ylim(0, 110)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% reference line')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot1_rejection_rates.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------
# Plot 2: Funnel chart - candidate flow through pipeline
# ---------------------------------------------------------------

def plot_funnel(aggregated, output_dir):
    """
    Horizontal funnel showing average candidate count at each stage.
    Shows how the pipeline reduces candidates step by step.

    Note: this is kept for quick visual sanity checks, but the main
    "flow story" plot is `plot_step_flow_log`.
    """

    stage_order = [
        'pnet_threshold',
        'pnet_nms_scale',
        'pnet_nms_cross',
        'rnet_threshold',
        'rnet_nms',
        'onet_threshold',
        'onet_nms',
    ]

    stage_labels = [
        'After P-Net Threshold',
        'After P-Net NMS (scale)',
        'After P-Net NMS (cross)',
        'After R-Net Threshold',
        'After R-Net NMS',
        'After O-Net Threshold',
        'After O-Net NMS (Final)',
    ]

    # Compute average input count for first stage (total proposals)
    first_stage_inputs = []
    for image_data in aggregated.values():
        if 'pnet_threshold' in image_data:
            first_stage_inputs.append(image_data['pnet_threshold']['input_count'])
    avg_initial = np.mean(first_stage_inputs) if first_stage_inputs else 0

    # Compute average output count at each stage
    avg_outputs = []
    for stage in stage_order:
        outputs = []
        for image_data in aggregated.values():
            if stage in image_data:
                outputs.append(image_data[stage]['output_count'])
        avg_outputs.append(np.mean(outputs) if outputs else 0)

    # Build funnel: initial count + output after each stage
    counts = [avg_initial] + avg_outputs
    labels = ['P-Net Input\n(all proposals)'] + stage_labels

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = [
        '#1565C0',
        '#2196F3', '#42A5F5', '#90CAF9',
        '#2E7D32', '#66BB6A',
        '#E65100', '#FF9800',
    ]

    y_positions = range(len(counts))

    counts_arr = np.asarray(counts, dtype=float)
    counts_for_width = np.log10(counts_arr + 1.0)
    max_count = float(np.max(counts_for_width)) if counts_for_width.size else 0.0

    for i, (count, label, color) in enumerate(zip(counts, labels, colors)):
        count_w = float(np.log10(float(count) + 1.0))
        width = ((count_w / max_count) * 0.8) if max_count > 0 else 0.0
        left = (1 - width) / 2

        ax.barh(
            i, width, left=left, height=0.6,
            color=color, edgecolor='black', linewidth=0.5
        )

        # Count label inside bar
        ax.text(
            0.5, i, f'{count:,.0f} boxes',
            ha='center', va='center',
            fontsize=9, fontweight='bold', color='white'
        )

        # Stage label on left
        ax.text(
            left - 0.02, i, label,
            ha='right', va='center', fontsize=9
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(counts) - 0.5)
    ax.axis('off')
    ax.set_title('Average Candidate Flow Through MTCNN Pipeline\n(across 1000 CelebA images)', fontsize=14)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot2_funnel.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------
# Plot 2b: Step plot - candidate flow through pipeline (log y-scale)
# ---------------------------------------------------------------

def plot_step_flow_log(aggregated, output_dir):
    """
    Step plot of average candidate counts across the pipeline, with log y-scale.

    This is usually clearer than a funnel when counts span orders of magnitude.
    """
    stage_order = [
        'pnet_threshold',
        'pnet_nms_scale',
        'pnet_nms_cross',
        'rnet_threshold',
        'rnet_nms',
        'onet_threshold',
        'onet_nms',
    ]

    stage_labels = [
        'P-In',
        'P-Th',
        'P-NMS(s)',
        'P-NMS(x)',
        'R-Th',
        'R-NMS',
        'O-Th',
        'O-NMS',
    ]

    # Average "input proposals" as the first point.
    first_stage_inputs = []
    for image_data in aggregated.values():
        if 'pnet_threshold' in image_data:
            first_stage_inputs.append(image_data['pnet_threshold']['input_count'])
    avg_initial = np.mean(first_stage_inputs) if first_stage_inputs else 0

    # Average output count after each decision point.
    avg_outputs = []
    for stage in stage_order:
        outputs = []
        for image_data in aggregated.values():
            if stage in image_data:
                outputs.append(image_data[stage]['output_count'])
        avg_outputs.append(np.mean(outputs) if outputs else 0)

    counts = np.asarray([avg_initial] + avg_outputs, dtype=float)

    # Avoid log(0): keep zeros visible at a floor, but show their true value in labels.
    plot_counts = np.maximum(counts, 1e-3)

    x = np.arange(len(plot_counts))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.step(x, plot_counts, where='post', linewidth=2.5, color='#1565C0')
    ax.scatter(x, plot_counts, s=45, color='#1565C0', zorder=3)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_ylabel('Average candidate count (log scale)')
    ax.set_xlabel('Pipeline decision point')
    ax.set_title('Average Candidate Flow Through MTCNN (log scale)\n(across 1000 CelebA images)')
    ax.grid(True, which='both', axis='y', alpha=0.25)

    # Annotate exact values (linear) for readability.
    for xi, c in zip(x, counts):
        ax.text(xi, max(float(np.maximum(c, 1e-3)), 1e-3) * 1.15, f'{c:,.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot2b_step_flow_log.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------
# Plot 3: Box plot - distribution of rejection rates
# ---------------------------------------------------------------

def plot_rejection_distribution(aggregated, output_dir):
    """
    Box plot showing the distribution of rejection rates per decision point.
    This shows consistency - not just the average but the spread.
    A decision with median=0 and small IQR is a strong redundancy candidate.
    """

    stage_order = [
        'pnet_threshold',
        'pnet_nms_scale',
        'pnet_nms_cross',
        'rnet_threshold',
        'rnet_nms',
        'onet_threshold',
        'onet_nms',
    ]

    stage_labels = [
        'P-Net\nThreshold',
        'P-Net NMS\n(per scale)',
        'P-Net NMS\n(cross scale)',
        'R-Net\nThreshold',
        'R-Net\nNMS',
        'O-Net\nThreshold',
        'O-Net\nNMS',
    ]

    all_rates = []
    for stage in stage_order:
        rates = []
        for image_data in aggregated.values():
            if stage in image_data:
                entry = image_data[stage]
                if entry['input_count'] > 0:
                    rate = entry['rejected_count'] / entry['input_count'] * 100
                    rates.append(rate)
        all_rates.append(rates)

    fig, ax = plt.subplots(figsize=(13, 6))

    colors = ['#2196F3'] * 3 + ['#4CAF50'] * 2 + ['#FF9800'] * 2

    bp = ax.boxplot(
        all_rates,
        labels=stage_labels,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=3, alpha=0.4)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title('Distribution of Rejection Rates per Decision Point\n(across 1000 CelebA images)', fontsize=14)
    ax.set_ylabel('Rejection Rate per Image (%)', fontsize=12)
    ax.set_xlabel('Decision Point', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add median labels above each box
    for i, rates in enumerate(all_rates):
        median = np.median(rates) if rates else 0
        ax.text(
            i + 1, median + 2,
            f'med={median:.1f}%',
            ha='center', va='bottom', fontsize=8
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color='#2196F3', alpha=0.7, label='P-Net decisions'),
        mpatches.Patch(color='#4CAF50', alpha=0.7, label='R-Net decisions'),
        mpatches.Patch(color='#FF9800', alpha=0.7, label='O-Net decisions'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot3_rejection_distribution.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------
# Plot 4: Heatmap - per-image rejection rates by decision point
# ---------------------------------------------------------------

def plot_rejection_heatmap(aggregated, output_dir, sort_rows=True):
    """
    Heatmap where each row is an image and each column is a decision point.
    Value is rejection rate (%) per image per decision.

    Missing stages or input_count==0 are shown as NaN.
    """
    stage_order = [
        'pnet_threshold',
        'pnet_nms_scale',
        'pnet_nms_cross',
        'rnet_threshold',
        'rnet_nms',
        'onet_threshold',
        'onet_nms',
    ]

    stage_labels = [
        'P-Th',
        'P-NMS(s)',
        'P-NMS(x)',
        'R-Th',
        'R-NMS',
        'O-Th',
        'O-NMS',
    ]

    image_ids = list(aggregated.keys())
    if not image_ids:
        print("No images found for heatmap; skipping.")
        return

    M = np.full((len(image_ids), len(stage_order)), np.nan, dtype=float)
    final_counts = np.full((len(image_ids),), np.nan, dtype=float)

    for i, image_id in enumerate(image_ids):
        img_data = aggregated.get(image_id, {})
        for j, stage in enumerate(stage_order):
            if stage not in img_data:
                continue
            entry = img_data[stage]
            inp = entry.get('input_count', 0)
            rej = entry.get('rejected_count', 0)
            if inp and inp > 0:
                M[i, j] = (rej / inp) * 100.0
        if 'onet_nms' in img_data:
            final_counts[i] = float(img_data['onet_nms'].get('output_count', np.nan))

    if sort_rows:
        # Sort: primary = final output count (ascending), secondary = total rejection (descending)
        total_rej = np.nansum(M, axis=1)
        final_key = np.nan_to_num(final_counts, nan=np.inf)
        order = np.lexsort((-total_rej, final_key))
        M = M[order, :]
        final_counts = final_counts[order]
        image_ids = [image_ids[k] for k in order]

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#444444')  # NaN / missing

    im = ax.imshow(M, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Rejection rate (%)')

    ax.set_xticks(np.arange(len(stage_labels)))
    ax.set_xticklabels(stage_labels, rotation=45, ha='right')

    # Too many y labels; show a few markers only.
    ax.set_yticks([])
    ax.set_ylabel(f'Images (n={M.shape[0]})')

    title = 'Per-Image Rejection Rates by Decision Point'
    if sort_rows:
        title += '\n(sorted by final O-NMS output, then total rejection)'
    ax.set_title(title)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot4_rejection_heatmap.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_plot.py logs/mtcnn_trace_XXXXXX.json")
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    output_dir = os.path.join(_PROJECT_ROOT, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading traces...")
    entries = load_traces(log_path)

    print("Grouping by image and stage...")
    aggregated = group_by_image_and_stage(entries)
    print(f"Images processed: {len(aggregated)}")

    print("\nGenerating plots...")
    plot_rejection_rates(aggregated, output_dir)
    # plot_funnel(aggregated, output_dir)  # optional (kept for sanity checks)
    plot_step_flow_log(aggregated, output_dir)
    plot_rejection_distribution(aggregated, output_dir)
    plot_rejection_heatmap(aggregated, output_dir, sort_rows=True)

    print("\nDone. All plots saved to outputs/")


if __name__ == "__main__":
    main()