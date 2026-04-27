"""
Computes dominance relationships between MTCNN decision points.
A decision X dominates decision Y if:
  - Whenever Y rejects a box, X also rejects it
  - Meaning Y never rejects anything X didn't already handle

Usage: python scripts/dominance_matrix.py logs/chunk4.jsonl
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_traces(log_path):
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'entries' in data:
        entries = data['entries']
    elif isinstance(data, list):
        entries = data
    else:
        print("ERROR: Unrecognized file format")
        sys.exit(1)
    
    print(f"Loaded {len(entries)} entries")
    return entries


def group_by_image(entries):
    """
    Group entries by image_id.
    For stages that appear multiple times (pnet_threshold, pnet_nms_scale),
    aggregate by summing input/output counts.
    Returns: dict of {image_id: {stage: {input_count, output_count, rejected_count}}}
    """
    raw = defaultdict(lambda: defaultdict(list))
    for e in entries:
        raw[e['image_id']][e['stage']].append(e)

    aggregated = {}
    for image_id, stages in raw.items():
        aggregated[image_id] = {}
        for stage, stage_entries in stages.items():
            if len(stage_entries) == 1:
                aggregated[image_id][stage] = {
                    'input_count':    stage_entries[0]['input_count'],
                    'output_count':   stage_entries[0]['output_count'],
                    'rejected_count': stage_entries[0]['rejected_count'],
                }
            else:
                aggregated[image_id][stage] = {
                    'input_count':    sum(e['input_count'] for e in stage_entries),
                    'output_count':   sum(e['output_count'] for e in stage_entries),
                    'rejected_count': sum(e['rejected_count'] for e in stage_entries),
                }
    return aggregated


def compute_rejection_vectors(aggregated):
    """
    For each decision point, compute a binary vector across all images.
    1 = this decision rejected at least one box on this image
    0 = this decision rejected nothing on this image
    Returns: dict of {stage: np.array of 0s and 1s}
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

    image_ids = sorted(aggregated.keys())
    vectors = {}

    for stage in stage_order:
        vec = []
        for image_id in image_ids:
            if stage in aggregated[image_id]:
                rejected = aggregated[image_id][stage]['rejected_count']
                vec.append(1 if rejected > 0 else 0)
            else:
                vec.append(0)
        vectors[stage] = np.array(vec)

    return vectors, image_ids


def compute_dominance_matrix(vectors):
    """
    Compute dominance matrix.

    dominance[X][Y] = percentage of images where:
      "Y rejected something" implies "X also rejected something"

    If dominance[X][Y] = 100%:
      X dominates Y completely
      Whenever Y does work, X already did that work
      Y may be redundant given X

    If dominance[X][Y] = 0%:
      X and Y are completely independent
    """
    stages = list(vectors.keys())
    n = len(stages)
    matrix = np.zeros((n, n))

    for i, stage_x in enumerate(stages):
        for j, stage_y in enumerate(stages):
            if i == j:
                matrix[i][j] = 100.0
                continue

            vec_x = vectors[stage_x]
            vec_y = vectors[stage_y]

            # Cases where Y rejected something
            y_rejected = vec_y == 1

            if y_rejected.sum() == 0:
                # Y never rejects anything → completely dominated by everything
                matrix[i][j] = 100.0
                continue

            # Of those cases, how often did X also reject something?
            x_also_rejected = (vec_x[y_rejected] == 1).sum()
            matrix[i][j] = x_also_rejected / y_rejected.sum() * 100

    return matrix, stages


def print_dominance_matrix(matrix, stages):
    """Print dominance matrix as a readable table."""

    short_names = {
        'pnet_threshold': 'P-Th',
        'pnet_nms_scale': 'P-NMS(s)',
        'pnet_nms_cross': 'P-NMS(x)',
        'rnet_threshold': 'R-Th',
        'rnet_nms':       'R-NMS',
        'onet_threshold': 'O-Th',
        'onet_nms':       'O-NMS',
    }

    labels = [short_names[s] for s in stages]

    print("\nDOMINANCE MATRIX")
    print("Row X, Col Y = % of images where Y rejecting implies X also rejected")
    print("100% = X completely dominates Y (Y may be redundant given X)")
    print("-" * 70)

    # Header
    print(f"{'':12}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print()

    print("-" * 70)

    for i, stage in enumerate(stages):
        print(f"{labels[i]:12}", end="")
        for j in range(len(stages)):
            val = matrix[i][j]
            print(f"{val:>9.1f}%", end="")
        print()

    print("-" * 70)


def plot_dominance_matrix(matrix, stages, output_dir):
    """Plot dominance matrix as a heatmap."""

    short_names = {
        'pnet_threshold': 'P-Th',
        'pnet_nms_scale': 'P-NMS(s)',
        'pnet_nms_cross': 'P-NMS(x)',
        'rnet_threshold': 'R-Th',
        'rnet_nms':       'R-NMS',
        'onet_threshold': 'O-Th',
        'onet_nms':       'O-NMS',
    }

    labels = [short_names[s] for s in stages]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add value labels inside each cell
    for i in range(len(stages)):
        for j in range(len(stages)):
            val = matrix[i][j]
            color = 'black' if 40 < val < 80 else 'white'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Dominance (%)')

    ax.set_xlabel('Decision Y (the one being dominated)', fontsize=11)
    ax.set_ylabel('Decision X (the dominator)', fontsize=11)
    ax.set_title(
        'Dominance Matrix\n'
        'Cell[X,Y] = % of images where Y rejecting implies X also rejected\n'
        'Green (100%) = X completely dominates Y',
        fontsize=11
    )

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'plot5_dominance_matrix.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def print_findings(matrix, stages, vectors):
    """Print human readable findings from the dominance matrix."""

    short_names = {
        'pnet_threshold': 'P-Net Threshold',
        'pnet_nms_scale': 'P-Net NMS scale',
        'pnet_nms_cross': 'P-Net NMS cross',
        'rnet_threshold': 'R-Net Threshold',
        'rnet_nms':       'R-Net NMS',
        'onet_threshold': 'O-Net Threshold',
        'onet_nms':       'O-Net NMS',
    }

    print("\nKEY FINDINGS")
    print("-" * 60)

    # Find decisions that never reject anything
    print("\n1. Decisions that NEVER reject anything:")
    found_any = False
    for stage, vec in vectors.items():
        if vec.sum() == 0:
            print(f"   {short_names[stage]}: rejected 0 boxes across ALL images")
            found_any = True
    if not found_any:
        print("   None found")

    # Find decisions that rarely reject anything
    print("\n2. Decisions that RARELY reject anything (< 10% of images):")
    found_any = False
    for stage, vec in vectors.items():
        rate = vec.sum() / len(vec) * 100
        if 0 < rate < 10:
            print(f"   {short_names[stage]}: active on only {rate:.1f}% of images")
            found_any = True
    if not found_any:
        print("   None found")

    # Find strong dominance relationships
    print("\n3. Strong dominance relationships (>= 95%):")
    found_any = False
    for i, stage_x in enumerate(stages):
        for j, stage_y in enumerate(stages):
            if i != j and matrix[i][j] >= 95:
                print(
                    f"   {short_names[stage_x]} dominates "
                    f"{short_names[stage_y]} "
                    f"({matrix[i][j]:.1f}%)"
                )
                found_any = True
    if not found_any:
        print("   None found")

    print("-" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/dominance_matrix.py logs/chunk4.jsonl")
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    output_dir = os.path.join(_PROJECT_ROOT, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    print("Loading traces...")
    entries = load_traces(log_path)

    print("Grouping by image...")
    aggregated = group_by_image(entries)
    print(f"Images: {len(aggregated)}")

    print("Computing rejection vectors...")
    vectors, image_ids = compute_rejection_vectors(aggregated)

    print("Computing dominance matrix...")
    matrix, stages = compute_dominance_matrix(vectors)

    print_dominance_matrix(matrix, stages)
    print_findings(matrix, stages, vectors)
    plot_dominance_matrix(matrix, stages, output_dir)

    print("\nDone. Dominance matrix saved to outputs/")


if __name__ == "__main__":
    main()