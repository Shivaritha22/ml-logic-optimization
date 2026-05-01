"""
Chunk 8: Benchmark original vs simplified MTCNN pipeline.

Measures:
  - Per-image wall clock time (perf_counter)
  - Total and mean speedup %
  - Throughput (images/sec)
  - cProfile stage breakdown (P-Net / R-Net / O-Net)

Plots generated (saved to plots/benchmark/):
  1. bar_speedup.png        - Headline mean time comparison
  2. boxplot_distribution.png - Per-image time distribution
  3. stacked_bar_stages.png  - P-Net / R-Net / O-Net time breakdown
  4. histogram_speedup.png   - Per-image speedup % distribution

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --max-images 500
"""

import sys
import os
import time
import json
import cProfile
import pstats
import io
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _PROJECT_ROOT)

from facenet_pytorch import MTCNN
from src.mtcnn.simplified_mtcnn import detect_face_simplified

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WIDER_VAL_DIR = os.path.join(
    _PROJECT_ROOT, 'data', 'widerface', 'WIDER_val', 'images'
)
BASELINE_PATH = os.path.join(_PROJECT_ROOT, 'baseline', 'wider_face_baseline.json')
PLOTS_DIR     = os.path.join(_PROJECT_ROOT, 'plots', 'benchmark')
RESULTS_PATH  = os.path.join(_PROJECT_ROOT, 'baseline', 'benchmark_results.json')

os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_image_paths(max_images=None):
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)
    keys = list(baseline.keys())
    if max_images:
        keys = keys[:max_images]
    paths = []
    for rel in keys:
        full = os.path.join(WIDER_VAL_DIR, rel)
        if os.path.exists(full):
            paths.append(full)
    return paths


def run_original(mtcnn, img):
    boxes, probs = mtcnn.detect(img)
    return boxes

def run_simplified(mtcnn, img):
    img_arr = np.array(img, dtype=np.float32)
    with torch.no_grad():
        boxes, _ = detect_face_simplified(
            img_arr,
            mtcnn.min_face_size,
            mtcnn.pnet,
            mtcnn.rnet,
            mtcnn.onet,
            mtcnn.thresholds,
            mtcnn.factor,
            torch.device('cpu'),
        )
    return boxes

# ─────────────────────────────────────────────
# Timing loop
# ─────────────────────────────────────────────
def time_pipeline(fn, mtcnn, image_paths, label):
    times = []
    print(f"\nTiming {label} on {len(image_paths)} images...")
    for i, path in enumerate(image_paths):
        if i % 500 == 0:
            print(f"  {i}/{len(image_paths)}")
        img = Image.open(path).convert('RGB') 
        t0 = time.perf_counter()
        fn(mtcnn, img)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)


# ─────────────────────────────────────────────
# cProfile stage breakdown
# ─────────────────────────────────────────────
def profile_pipeline(fn, mtcnn, image_paths, label, n_profile=100):
    """
    Profile on first n_profile images.
    Returns a dict of {function_name: stagetime} for key MTCNN internals.
    """
    print(f"\nProfiling {label} (first {n_profile} images)...")
    pr = cProfile.Profile()
    pr.enable()
    for path in image_paths[:n_profile]:
        img = Image.open(path).convert('RGB')
        fn(mtcnn, img)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats('cumulative')
    ps.print_stats(30)
    profile_text = stream.getvalue()

    # Extract stagetime for key stage functions
    # Adjust these names to match your actual module paths if needed
    stage_keywords = {
        'P-Net': ['pnet', 'generate_bboxes', 'detect_face_pnet'],
        'R-Net': ['rnet', 'detect_face_rnet'],
        'O-Net': ['onet', 'detect_face_onet'],
        'NMS':   ['nms', 'non_max_suppression'],
    }

    stage_times = {k: 0.0 for k in stage_keywords}
    for line in profile_text.split('\n'):
        for stage, keywords in stage_keywords.items():
            if any(kw in line.lower() for kw in keywords):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        stagetime = float(parts[3])
                        if stagetime > stage_times[stage]:
                            stage_times[stage] = stagetime
                    except ValueError:
                        pass

    # Save raw profile text
    profile_out = os.path.join(PLOTS_DIR, f'profile_{label.lower().replace(" ", "_")}.txt')
    with open(profile_out, 'w') as f:
        f.write(profile_text)
    print(f"  Full profile saved to {profile_out}")

    return stage_times, profile_text


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────
def plot_bar_speedup(orig_times, simp_times):
    orig_mean = np.mean(orig_times) * 1000  # ms
    simp_mean = np.mean(simp_times) * 1000
    speedup   = (orig_mean - simp_mean) / orig_mean * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ['Original MTCNN', 'Simplified MTCNN'],
        [orig_mean, simp_mean],
        color=['#4C72B0', '#55A868'],
        width=0.5,
        edgecolor='black',
        linewidth=0.8
    )
    for bar, val in zip(bars, [orig_mean, simp_mean]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.1f} ms',
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )

    ax.set_ylabel('Mean time per image (ms)', fontsize=12)
    ax.set_title(f'Original vs Simplified — {speedup:.1f}% speedup', fontsize=13)
    ax.set_ylim(0, orig_mean * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'bar_speedup.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_boxplot_distribution(orig_times, simp_times):
    data = [orig_times * 1000, simp_times * 1000]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        data,
        labels=['Original', 'Simplified'],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2)
    )
    colors = ['#4C72B0', '#55A868']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Time per image (ms)', fontsize=12)
    ax.set_title('Per-image time distribution', fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'boxplot_distribution.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_histogram_speedup(orig_times, simp_times):
    per_image_speedup = (orig_times - simp_times) / orig_times * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_image_speedup, bins=50, color='#C44E52', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axvline(np.mean(per_image_speedup), color='black', linestyle='--', linewidth=1.5,
               label=f'Mean: {np.mean(per_image_speedup):.1f}%')
    ax.set_xlabel('Speedup per image (%)', fontsize=12)
    ax.set_ylabel('Number of images', fontsize=12)
    ax.set_title('Distribution of per-image speedup', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'histogram_speedup.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_stacked_bar_stages(orig_stages, simp_stages):
    stages = [s for s in orig_stages if orig_stages[s] > 0 or simp_stages[s] > 0]
    if not stages:
        print("  Warning: No stage data extracted from cProfile — skipping stacked bar.")
        return

    x      = np.arange(2)
    width  = 0.5
    bottom_orig = 0
    bottom_simp = 0
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(8, 5))
    for stage, color in zip(stages, colors):
        orig_val = orig_stages[stage]
        simp_val = simp_stages[stage]
        ax.bar(0, orig_val, width, bottom=bottom_orig, label=stage, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        ax.bar(1, simp_val, width, bottom=bottom_simp, color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        bottom_orig += orig_val
        bottom_simp += simp_val

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Original', 'Simplified'], fontsize=12)
    ax.set_ylabel('Cumulative time (s)', fontsize=12)
    ax.set_title('Stage breakdown — P-Net / R-Net / O-Net / NMS', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'stacked_bar_stages.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-images', type=int, default=None,
                        help='Limit number of images (default: all 3226)')
    args = parser.parse_args()

    print("CHUNK 8: Benchmarking Original vs Simplified MTCNN")
    print("=" * 60)

    device = torch.device('cpu')
    mtcnn  = MTCNN(keep_all=True, device=device)

    image_paths = load_image_paths(args.max_images)
    print(f"Images to benchmark: {len(image_paths)}")

    # ── Timing ──
    orig_times = []
    simp_times = []
    print(f"\nInterleaved timing on {len(image_paths)} images...")
    for i, path in enumerate(image_paths):
        if i % 500 == 0:
            print(f"  {i}/{len(image_paths)}")
        img = Image.open(path).convert('RGB')
    
        t0 = time.perf_counter()
        run_original(mtcnn, img)
        t1 = time.perf_counter()
        orig_times.append(t1 - t0)

        t0 = time.perf_counter()
        run_simplified(mtcnn, img)
        t1 = time.perf_counter()
        simp_times.append(t1 - t0)

    orig_times = np.array(orig_times)
    simp_times = np.array(simp_times)

    # ── Stats ──
    orig_mean  = np.mean(orig_times) * 1000
    simp_mean  = np.mean(simp_times) * 1000
    speedup    = (orig_mean - simp_mean) / orig_mean * 100
    throughput_orig = len(image_paths) / orig_times.sum()
    throughput_simp = len(image_paths) / simp_times.sum()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Images benchmarked   : {len(image_paths)}")
    print(f"Original  mean time  : {orig_mean:.2f} ms/image")
    print(f"Simplified mean time : {simp_mean:.2f} ms/image")
    print(f"Speedup              : {speedup:.2f}%")
    print(f"Original  throughput : {throughput_orig:.2f} img/s")
    print(f"Simplified throughput: {throughput_simp:.2f} img/s")
    print(f"Orig  p50/p95/p99    : {np.percentile(orig_times*1000,50):.1f} / "
          f"{np.percentile(orig_times*1000,95):.1f} / {np.percentile(orig_times*1000,99):.1f} ms")
    print(f"Simp  p50/p95/p99    : {np.percentile(simp_times*1000,50):.1f} / "
          f"{np.percentile(simp_times*1000,95):.1f} / {np.percentile(simp_times*1000,99):.1f} ms")

    # ── cProfile stage breakdown ──
    orig_stages, _ = profile_pipeline(run_original,   mtcnn, image_paths, "Original")
    simp_stages, _ = profile_pipeline(run_simplified, mtcnn, image_paths, "Simplified")

    print("\nStage breakdown (stagetime, first 100 images):")
    print(f"  {'Stage':<10} {'Original':>12} {'Simplified':>12}")
    for stage in orig_stages:
        print(f"  {stage:<10} {orig_stages[stage]:>11.3f}s {simp_stages[stage]:>11.3f}s")

    # ── Save results JSON ──
    results = {
        "n_images"          : len(image_paths),
        "orig_mean_ms"      : round(orig_mean, 3),
        "simp_mean_ms"      : round(simp_mean, 3),
        "speedup_pct"       : round(speedup, 3),
        "throughput_orig"   : round(throughput_orig, 3),
        "throughput_simp"   : round(throughput_simp, 3),
        "orig_p50_ms"       : round(float(np.percentile(orig_times*1000, 50)), 3),
        "orig_p95_ms"       : round(float(np.percentile(orig_times*1000, 95)), 3),
        "orig_p99_ms"       : round(float(np.percentile(orig_times*1000, 99)), 3),
        "simp_p50_ms"       : round(float(np.percentile(simp_times*1000, 50)), 3),
        "simp_p95_ms"       : round(float(np.percentile(simp_times*1000, 95)), 3),
        "simp_p99_ms"       : round(float(np.percentile(simp_times*1000, 99)), 3),
        "orig_stage_times_s": {k: round(v, 4) for k, v in orig_stages.items()},
        "simp_stage_times_s": {k: round(v, 4) for k, v in simp_stages.items()},
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_bar_speedup(orig_times, simp_times)
    plot_boxplot_distribution(orig_times, simp_times)
    plot_histogram_speedup(orig_times, simp_times)
    plot_stacked_bar_stages(orig_stages, simp_stages)

    print("\n" + "=" * 60)
    print(f"✅ Chunk 8 complete — {speedup:.1f}% speedup")
    print(f"   Plots in : plots/benchmark/")
    print(f"   Results  : baseline/benchmark_results.json")
    print("=" * 60)


if __name__ == '__main__':
    main()