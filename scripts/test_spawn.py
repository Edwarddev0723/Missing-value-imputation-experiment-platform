"""Quick test: spawn workers with real experiment configs."""
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

    # Find 20 pending experiments
    pending = []
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
            pending.append(cfg)
            if len(pending) >= 20:
                break

    print(f"Testing {len(pending)} pending experiments with 4 workers...")
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=4, mp_context=ctx,
        initializer=partial(_worker_init, 4),
    ) as executor:
        futures = {executor.submit(_worker_run_experiment, c): i for i, c in enumerate(pending)}
        done = 0
        for f in as_completed(futures):
            try:
                r = f.result(timeout=300)
                done += 1
                print(f"  [{done}/{len(pending)}] {r['status']}: {r['tag']} ({time.time()-t0:.1f}s)")
            except Exception as e:
                done += 1
                print(f"  [{done}/{len(pending)}] ERROR: {type(e).__name__}: {e}")
    print(f"\nDone in {time.time()-t0:.1f}s")
