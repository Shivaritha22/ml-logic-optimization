# MTCNN Boolean Simplification Pipeline

This project profiles an MTCNN face detection pipeline, identifies redundant boolean decision logic, simplifies the pipeline, validates the simplified version against a saved baseline, and formally verifies the optimization using Rust + egg.

The goal is to show that a profiling-guided optimization can be:

- discovered from real execution traces
- simplified using boolean algebra
- formally verified
- validated against baseline outputs
- benchmarked for runtime impact

---

## Table of Contents

- [Requirements](#requirements)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Step 1: Smoke Test](#step-1-smoke-test)
- [Step 2: Trace and Profile MTCNN](#step-2-trace-and-profile-mtcnn)
- [Step 3: Analyze Decision Points](#step-3-analyze-decision-points)
- [Step 4: Save Baseline](#step-4-save-baseline)
- [Step 5: Validate Simplified Pipeline](#step-5-validate-simplified-pipeline)
- [Step 6: Formal Verification](#step-6-formal-verification)
- [Step 7: Benchmark](#step-7-benchmark)
- [Full Command Order](#full-command-order)
- [Final Correctness Argument](#final-correctness-argument)

---

## Requirements

Install the Python dependencies:

```bash
pip install facenet-pytorch torch torchvision numpy pandas matplotlib pillow
```

Install Rust and Cargo:

```text
https://rustup.rs
```

No additional Rust dependencies need to be installed manually. The `egg` dependency is fetched automatically by Cargo.

---

## Datasets

The project expects the following datasets:

```text
CelebA:
img_align_celeba/

WIDER FACE validation set:
WIDER_val/images/
```

---

## Project Structure

```text
ml-logic-optimization/
├── baseline/                        # saved outputs and benchmark results
├── boolean_simplifier/              # Rust egg prover and exhaustive verifier
│   ├── src/
│   │   ├── analysis.py              # parses trace and identifies redundant decisions
│   │   ├── main.rs                  # egg equality saturation proof
│   │   └── verify.rs                # exhaustive equivalence verifier
│   ├── main Cargo.toml              # simplify binary config
│   └── verify Cargo.toml            # verify binary config
├── data/                            # CelebA and WIDER FACE datasets
├── docs/                            # decision point map
├── logs/                            # MTCNN trace files from profiling
├── outputs/                         # validation results
├── plots/                           # all generated charts
├── scripts/
│   ├── analyze_plot.py              # profiling plots and dominance matrix
│   ├── baseline.py                  # saves original pipeline outputs
│   ├── dominance_matrix.py          # computes decision dominance relationships
│   ├── mtcnn_benchmark.py           # benchmarks original vs simplified
│   └── run_profiling.py             # instruments and traces MTCNN
├── src/
│   ├── mtcnn/                       # original and simplified pipeline
│   ├── profiling/                   # trace wrapper and instrumentation
│   └── validation/                  # validator comparing outputs to baseline
└── tests/
    ├── smoke_test.py                # end-to-end MTCNN smoke test
    └── smoke_simplified_mtcnn.py    # smoke test for simplified pipeline
```
---

## Pipeline Overview

The pipeline has seven main stages:

```text
1. Smoke test
2. Trace and profile MTCNN
3. Analyze decision points
4. Save baseline
5. Validate simplified pipeline
6. Formally verify optimization
7. Benchmark performance
```

---

## Step 1: Smoke Test

Run:

```bash
python tests/smoke_test.py wider
```

This checks that:

- the environment is configured correctly
- the WIDER FACE dataset path is valid
- the MTCNN pipeline runs end-to-end without crashing

---

## Step 2: Trace and Profile MTCNN

Run:

```bash
python scripts/trace_mtcnn.py
```

This generates a trace file:

```text
logs/mtcnn_trace_<timestamp>.json
```

The trace records runtime boolean decision points inside the MTCNN pipeline.

---

## Step 3: Analyze Decision Points

Run:

```bash
python boolean_simplifier/src/analysis.py logs/mtcnn_trace_<timestamp>.json
```

Replace `<timestamp>` with the actual generated trace filename.

This generates:

```text
boolean_simplifier/src/main.rs
```

This step:

- reads the traced MTCNN decision data
- identifies redundant boolean checks
- generates Rust code for simplification and verification

---

## Step 4: Save Baseline

Run:

```bash
python scripts/save_baseline.py
```

This generates:

```text
baseline/wider_face_baseline.json
```

The baseline stores the original MTCNN output before simplification. It is used as the reference output during validation.

---

## Step 5: Validate Simplified Pipeline

Run:

```bash
python src/validation/validator.py
```

This compares the simplified pipeline against the saved baseline.

This proves that the simplified pipeline preserves the original face detection behavior.

---

## Step 6: Formal Verification

Formal verification is done using Rust + egg.

Navigate to the boolean simplifier directory:

```bash
cd boolean_simplifier
```

Make sure `Cargo.toml` is configured with both binaries:

```text
simplify
verify
```

---

### Step 6.1: Run the egg Simplifier

Run:

```bash
cargo run --bin simplify
```

Expected output:

```text
Input:  (and A (and B (and true (and D (and E (and F G))))))
Output: (and A (and B (and D (and E (and F G)))))
```

This proves the boolean simplification algebraically:

```text
A AND B AND TRUE AND D AND E AND F AND G
=
A AND B AND D AND E AND F AND G
```

The rewrite is valid because of the boolean identity:

```text
X AND TRUE = X
```

So the redundant `TRUE` decision can be safely removed.

---

### Step 6.2: Run the Exhaustive Verifier

Run:

```bash
cargo run --bin verify
```

Expected output:

```text
RESULT: VERIFIED
All 64 inputs (C=TRUE) produced identical decisions.
Optimization is both formally proven and exhaustively verified.
```

The verifier checks all valid input combinations under the profiling constraint:

```text
C = TRUE
```

Since there are 6 variable boolean inputs:

```text
2^6 = 64
```

the verifier exhaustively checks all 64 cases and confirms that the original and simplified pipelines produce identical decisions.

---

### What Formal Verification Proves

The formal verification has two parts:

```text
simplify:
Uses egg equality saturation to prove the boolean rewrite algebraically.

verify:
Exhaustively enumerates all valid boolean inputs under C = TRUE and checks that the original and simplified decisions match.
```

Together, they show that the optimization is both algebraically correct and computationally verified.

---

## Step 7: Benchmark

Return to the project root:

```bash
cd ..
```

Run a small benchmark first:

```bash
python scripts/mtcnn_benchmark.py --max-images 100
```

Then run the full benchmark:

```bash
python scripts/mtcnn_benchmark.py
```

Benchmark plots are generated in:

```text
plots/benchmark/
```

The benchmark measures the runtime difference between the original and simplified pipeline.

---

## Full Command Order

```bash
pip install facenet-pytorch torch torchvision numpy pandas matplotlib pillow

python tests/smoke_test.py wider

python scripts/trace_mtcnn.py

python boolean_simplifier/src/analysis.py logs/mtcnn_trace_<timestamp>.json

python scripts/save_baseline.py

python src/validation/validator.py

cd boolean_simplifier

cargo run --bin simplify

cargo run --bin verify

cd ..

python scripts/mtcnn_benchmark.py --max-images 100

python scripts/mtcnn_benchmark.py
```

---

## Final Correctness Argument

The original MTCNN pipeline is first traced on real face detection data. The trace identifies a boolean decision point that is always `TRUE` under the profiling constraint. The boolean expression is then simplified by removing this redundant condition.

The optimization is verified in three ways:

```text
1. Algebraic proof:
   egg proves that X AND TRUE = X.

2. Exhaustive verification:
   all 64 valid input combinations under C = TRUE are checked.

3. Baseline validation:
   the simplified pipeline output is compared against the saved original MTCNN baseline.
```

Therefore, the simplified MTCNN pipeline is behavior-preserving, formally verified, validated against the baseline, and benchmarked for runtime impact.

---


