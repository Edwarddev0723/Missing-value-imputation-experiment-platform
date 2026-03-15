#!/usr/bin/env python3
"""
Master pipeline runner: executes all analysis steps in sequence.
Usage: python analysis/scripts/run_all.py
"""
import sys
import os
import time

# Ensure analysis/scripts is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_step(module_name, step_name):
    print(f"\n{'#' * 70}")
    print(f"# {step_name}")
    print(f"{'#' * 70}")
    t0 = time.time()
    mod = __import__(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"[{step_name}] completed in {elapsed:.1f}s")
    return elapsed


def main():
    t_start = time.time()
    
    steps = [
        ("step1_consolidate", "Step 1: Data Consolidation & Validation"),
        ("step2_audit", "Step 2: Exploratory Audit"),
        ("step3_analysis", "Steps 3-4: Stratified & Track Analysis"),
        ("step5_stats", "Step 5: Statistical Testing"),
        ("step6_visualize", "Step 6: Visualizations"),
        ("step7_tables", "Step 7: Paper-Ready Tables"),
        ("step8_conclusions", "Step 8: Auto-Generated Conclusions"),
    ]
    
    timings = {}
    for module, name in steps:
        try:
            elapsed = run_step(module, name)
            timings[name] = elapsed
        except Exception as e:
            print(f"\n*** ERROR in {name}: {type(e).__name__}: {e} ***")
            import traceback
            traceback.print_exc()
            timings[name] = f"FAILED: {e}"
    
    total = time.time() - t_start
    
    print(f"\n{'=' * 70}")
    print(f"Pipeline Complete — Total: {total:.1f}s")
    print(f"{'=' * 70}")
    for name, t in timings.items():
        if isinstance(t, float):
            print(f"  {name}: {t:.1f}s")
        else:
            print(f"  {name}: {t}")
    
    # Print output file list
    from paths import OUTPUTS_DIR
    print(f"\nOutput files:")
    for root, dirs, files in os.walk(OUTPUTS_DIR):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            rel = os.path.relpath(path, OUTPUTS_DIR)
            print(f"  {rel} ({size:,} bytes)")


if __name__ == "__main__":
    main()
