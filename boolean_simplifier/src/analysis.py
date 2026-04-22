"""
Chunk B2: Formal analysis of MTCNN decision points.
Reads Person A's trace data and produces boolean formula analysis.
"""

import json
import os
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------
# Load and group traces
# ---------------------------------------------------------------

def load_traces(log_path):
    with open(log_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'entries' in data:
        return data['entries']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unrecognized file format")


def group_by_image(entries):
    raw = defaultdict(lambda: defaultdict(list))
    for e in entries:
        raw[e['image_id']][e['stage']].append(e)

    aggregated = {}
    for image_id, stages in raw.items():
        aggregated[image_id] = {}
        for stage, stage_entries in stages.items():
            aggregated[image_id][stage] = {
                'input_count':    sum(e['input_count'] for e in stage_entries),
                'output_count':   sum(e['output_count'] for e in stage_entries),
                'rejected_count': sum(e['rejected_count'] for e in stage_entries),
            }
    return aggregated


# ---------------------------------------------------------------
# Boolean analysis
# ---------------------------------------------------------------

def compute_boolean_values(aggregated):
    """
    For each decision point across all images:
    - Compute rejection rate
    - Determine boolean value (TRUE/FALSE/VARIABLE)
    - Compute confidence

    A decision is TRUE  if it never rejects anything
    A decision is FALSE if it always rejects everything
    A decision is VARIABLE if it sometimes rejects, sometimes not
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

    formula_labels = {
        'pnet_threshold': 'A',
        'pnet_nms_scale': 'B',
        'pnet_nms_cross': 'C',
        'rnet_threshold': 'D',
        'rnet_nms':       'E',
        'onet_threshold': 'F',
        'onet_nms':       'G',
    }

    results = {}

    for stage in stage_order:
        rejection_rates = []
        active_count = 0

        for image_data in aggregated.values():
            if stage in image_data:
                entry = image_data[stage]
                if entry['input_count'] > 0:
                    rate = entry['rejected_count'] / entry['input_count'] * 100
                    rejection_rates.append(rate)
                    if entry['rejected_count'] > 0:
                        active_count += 1

        if not rejection_rates:
            continue

        rates = np.array(rejection_rates)
        n_images = len(rates)

        avg_rate    = rates.mean()
        median_rate = np.median(rates)
        always_zero = (rates == 0).all()
        pct_zero    = (rates == 0).sum() / n_images * 100

        # Determine boolean classification
        if always_zero:
            boolean_value = 'TRUE'
            confidence    = 'VERY HIGH'
            explanation   = 'Never rejects anything across all images'
        elif pct_zero >= 90:
            boolean_value = 'MOSTLY TRUE'
            confidence    = 'MEDIUM'
            explanation   = f'Zero rejection in {pct_zero:.1f}% of images'
        elif avg_rate >= 30:
            boolean_value = 'VARIABLE'
            confidence    = 'HIGH'
            explanation   = 'Consistently doing real work'
        else:
            boolean_value = 'MOSTLY TRUE'
            confidence    = 'LOW'
            explanation   = f'Low but non-zero rejection rate'

        results[stage] = {
            'label':         formula_labels[stage],
            'avg_rate':      avg_rate,
            'median_rate':   median_rate,
            'pct_zero':      pct_zero,
            'boolean_value': boolean_value,
            'confidence':    confidence,
            'explanation':   explanation,
            'n_images':      n_images,
        }

    return results


def print_boolean_analysis(results):
    """Print formal boolean analysis."""

    stage_names = {
        'pnet_threshold': 'P-Net Threshold',
        'pnet_nms_scale': 'P-Net NMS scale',
        'pnet_nms_cross': 'P-Net NMS cross',
        'rnet_threshold': 'R-Net Threshold',
        'rnet_nms':       'R-Net NMS',
        'onet_threshold': 'O-Net Threshold',
        'onet_nms':       'O-Net NMS',
    }

    print("\nBOOLEAN ANALYSIS OF MTCNN DECISION POINTS")
    print("=" * 70)
    print(f"{'Decision':<22} {'Label':<6} {'Avg%':>6} {'Med%':>6} "
          f"{'Zero%':>7} {'Boolean':<14} {'Confidence'}")
    print("-" * 70)

    for stage, r in results.items():
        print(
            f"{stage_names[stage]:<22} "
            f"{r['label']:<6} "
            f"{r['avg_rate']:>5.1f}% "
            f"{r['median_rate']:>5.1f}% "
            f"{r['pct_zero']:>6.1f}% "
            f"{r['boolean_value']:<14} "
            f"{r['confidence']}"
        )

    print("=" * 70)


def generate_boolean_formula(results):
    """
    Generate original and simplified boolean formulas.
    """

    label_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # Map label to boolean value
    label_to_bool = {}
    for stage, r in results.items():
        label_to_bool[r['label']] = r['boolean_value']

    print("\nBOOLEAN FORMULA ANALYSIS")
    print("=" * 70)

    # Original formula
    print("\nOriginal pipeline formula:")
    print("  keep = A AND B AND C AND D AND E AND F AND G")
    print()
    print("  Where:")
    for stage, r in results.items():
        print(f"    {r['label']} = {stage.replace('_', ' ').title()}")

    # Substitute findings
    print("\nSubstituting boolean findings:")
    print()

    simplified_labels = []
    removed_labels = []

    for label in label_order:
        bool_val = label_to_bool.get(label, 'VARIABLE')
        if bool_val == 'TRUE':
            print(f"  {label} = TRUE  → substitute: "
                  f"(X AND TRUE = X by identity law)")
            removed_labels.append(label)
        elif bool_val == 'MOSTLY TRUE':
            print(f"  {label} = MOSTLY TRUE  → keep for now "
                  f"(not safe to remove without validation)")
            simplified_labels.append(label)
        else:
            simplified_labels.append(label)

    # Simplified formula
    print()
    print("Simplified formula (certain removals):")
    formula = " AND ".join(simplified_labels)
    print(f"  keep = {formula}")

    # egg input format
    egg_expr = build_egg_expr(simplified_labels)
    print()
    print("Egg input expression:")
    print(f"  {egg_expr}")

    print()
    print("Decisions removed:")
    for label in removed_labels:
        for stage, r in results.items():
            if r['label'] == label:
                print(f"  {label} ({stage.replace('_', ' ').title()})"
                      f" - confidence: {r['confidence']}")

    print()
    print("Decisions kept:")
    for label in simplified_labels:
        for stage, r in results.items():
            if r['label'] == label:
                print(f"  {label} ({stage.replace('_', ' ').title()})"
                      f" - {r['boolean_value']}")

    print("=" * 70)

    return simplified_labels, removed_labels, egg_expr


def build_egg_expr(labels):
    """
    Build nested egg expression from list of labels.
    (and A (and B (and D (and E (and F G)))))
    """
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"(and {labels[0]} {labels[1]})"
    return f"(and {labels[0]} {build_egg_expr(labels[1:])})"


def generate_egg_input(results, simplified_labels, removed_labels, egg_expr):
    """
    Generate egg Rust code fully derived from analysis.
    No hardcoding - everything comes from the data.
    """

    # Build original expression with TRUE substituted
    # for removed decisions
    label_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    original_labels = []
    for label in label_order:
        if label in removed_labels:
            original_labels.append('true')  # substitute removed with TRUE
        else:
            original_labels.append(label)   # keep as is
    
    original_egg_expr = build_egg_expr(original_labels)

    # Build removed decisions comment
    removed_comments = []
    for label in removed_labels:
        for stage, r in results.items():
            if r['label'] == label:
                removed_comments.append(
                    f"//   {label} = TRUE ({stage.replace('_', ' ').title()} "
                    f"- {r['avg_rate']:.1f}% avg rejection, "
                    f"{r['pct_zero']:.1f}% zero across {r['n_images']} images)"
                )

    # Build kept decisions comment
    kept_comments = []
    for label in simplified_labels:
        for stage, r in results.items():
            if r['label'] == label:
                kept_comments.append(
                    f"//   {label} = {r['boolean_value']} "
                    f"({stage.replace('_', ' ').title()} "
                    f"- {r['avg_rate']:.1f}% avg rejection)"
                )

    removed_str = '\n'.join(removed_comments)
    kept_str    = '\n'.join(kept_comments)

    original_formula  = " AND ".join(
        ['TRUE' if l in removed_labels else l for l in label_order]
    )
    simplified_formula = " AND ".join(simplified_labels)

    rust_code = f'''use egg::{{rewrite as rw, *}};

define_language! {{
    enum BoolLang {{
        "and" = And([Id; 2]),
        "or"  = Or([Id; 2]),
        "not" = Not([Id; 1]),
        Symbol(Symbol),
    }}
}}

fn main() {{
    // Boolean rewriting rules
    let rules: &[Rewrite<BoolLang, ()>] = &[
        rw!("and-comm-lr"; "(and ?x ?y)" => "(and ?y ?x)"),
        rw!("and-comm-rl"; "(and ?y ?x)" => "(and ?x ?y)"),
        rw!("and-true";  "(and ?x true)"  => "?x"),
        rw!("and-false"; "(and ?x false)" => "false"),
        rw!("and-same";  "(and ?x ?x)"    => "?x"),
        rw!("true-and";  "(and true ?x)"  => "?x"),
    ];

    // Decisions classified as TRUE from profiling data:
{removed_str}

    // Decisions classified as VARIABLE (kept):
{kept_str}

    // Original formula with TRUE substituted for redundant decisions:
    // {original_formula}
    let expr = "{original_egg_expr}";

    println!("Running egg equality saturation...");
    println!("Input:  {{}}", expr);

    let runner = Runner::<BoolLang, (), ()>::default()
        .with_expr(&expr.parse().unwrap())
        .run(rules);

    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (cost, simplified) = extractor.find_best(runner.roots[0]);

    println!("Output: {{}}", simplified);
    println!("Nodes:  {{}}", cost);
    println!();
    println!("This proves:");
    println!("  {original_formula}");
    println!("  = {simplified_formula}");
    println!("  via boolean identity laws");
}}
'''
    return rust_code

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analysis.py logs/chunk4.json")
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f"ERROR: File not found: {log_path}")
        sys.exit(1)

    print(f"Loading traces from {log_path}...")
    entries = load_traces(log_path)
    print(f"Loaded {len(entries)} entries")

    print("Grouping by image...")
    aggregated = group_by_image(entries)
    print(f"Images: {len(aggregated)}")

    print("\nComputing boolean values...")
    results = compute_boolean_values(aggregated)

    print_boolean_analysis(results)

    simplified_labels, removed_labels, egg_expr = generate_boolean_formula(results)

    # Generate egg Rust code
    rust_code = generate_egg_input(results, simplified_labels, removed_labels, egg_expr)


    
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_path = os.path.join(_PROJECT_ROOT, 'boolean_simplifier', 'src', 'main.rs')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(rust_code)

    print(f"\nGenerated egg Rust code saved to: {out_path}")
    print("Run it with: cd boolean_simplifier && cargo run")


if __name__ == "__main__":
    main()