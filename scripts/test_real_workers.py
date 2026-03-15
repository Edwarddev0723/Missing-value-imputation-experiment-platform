"""Test real _worker_run_experiment with spawn context."""
import sys, os
sys.path.insert(0, ".")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["WANDB_MODE"] = "disabled"

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial
    from pathlib import Path
    import time

    from src.config import load_yaml, expand_batch_config
    from src.orchestration.batch_runner import _worker_run_experiment, _worker_init

    batch_config = load_yaml("configs/experiments/full_benchmark_nondl.yaml")
    configs = expand_batch_config(batch_config, "configs")

    # Find 8 pending experiments (pick different methods)
    pending = []
    seen_methods = set()
    for cfg in configs:
        ds = cfg.get("dataset",{}).get("name","?")
        method = cfg.get("method",{}).get("name","?")
        mechanism = cfg.get("missingness",{}).get("mechanism","?")
        pattern = cfg.get("missingness",{}).get("pattern","general")
        rate = cfg.get("missingness",{}).get("rate",0.0)
        seed = cfg.get("project",{}).get("seed",0)
        rate_pct = int(round(rate * 100))
        eid = f"{ds}_{method}_{mechanism}_{pattern}_{rate_pct}pct_s{seed}"
        if not (Path("results") / "raw" / f"{eid}.json").exists():
            if method not in seen_methods:
                pending.append((cfg, eid))
                seen_methods.add(method)
                print(f"  Will test: {eid}")
            if len(pending) >= 8:
                break

    print(f"\nRunning {len(pending)} experiments with 4 workers (spawn)...")
    ctx = mp.get_context("spawn")
    max_w = 4
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=max_w, mp_context=ctx,
        initializer=partial(_worker_init, max_w),
    ) as executor:
        futures = {}
        for i, (cfg, eid) in enumerate(pending):
            futures[executor.submit(_worker_run_experiment, cfg)] = eid

        done = 0
        for f in as_completed(futures):
            eid = futures[f]
            done += 1
            try:
                r = f.result(timeout=300)
                elapsed = time.time() - t0
                print(f"  [{done}/{len(pending)}] {r['status']}: {r['tag']} ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{done}/{len(pending)}] ERROR: {type(e).__name__}: {e} ({elapsed:.1f}s)")

    print(f"\nAll done in {time.time()-t0:.1f}s")
