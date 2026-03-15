"""Check pending experiments and diagnose hang."""
import sys, os
sys.path.insert(0, ".")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from pathlib import Path
from collections import Counter
from src.config import load_yaml, expand_batch_config

batch_config = load_yaml("configs/experiments/full_benchmark_nondl.yaml")
configs = expand_batch_config(batch_config, "configs")

skip_count = 0
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
    result_path = Path("results") / "raw" / f"{eid}.json"
    if result_path.exists():
        skip_count += 1
    else:
        pending.append((eid, method, ds))

print(f"Total: {len(configs)}, Skip: {skip_count}, Pending: {len(pending)}")
method_counts = Counter(m for _,m,_ in pending)
print(f"Pending by method: {dict(method_counts)}")
ds_counts = Counter(d for _,_,d in pending)
print(f"Pending by dataset: {dict(ds_counts)}")

print("\nFirst 30 pending:")
for eid, method, ds in pending[:30]:
    print(f"  {eid}")

# Now test: can we actually run a single experiment in a spawn worker?
print("\n=== Test: single experiment in spawn worker ===")
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time

from src.orchestration.batch_runner import _worker_run_experiment, _worker_init

if __name__ == "__main__":
    # Pick a small pending experiment (mean method, small dataset)
    test_cfg = None
    for cfg in configs:
        ds = cfg.get("dataset",{}).get("name","?")
        method = cfg.get("method",{}).get("name","?")
        eid_check = f"{ds}_{method}"
        result_path = Path("results") / "raw"
        # Pick a mean imputer experiment that's not done
        if method == "mean" and not any(result_path.glob(f"{ds}_mean_*.json")):
            test_cfg = cfg
            break
    
    if test_cfg is None:
        # Just pick any pending
        for cfg in configs:
            ds = cfg.get("dataset",{}).get("name","?")
            method = cfg.get("method",{}).get("name","?")
            rate_pct = int(round(cfg.get("missingness",{}).get("rate",0)*100))
            seed = cfg.get("project",{}).get("seed",0)
            eid = f"{ds}_{method}_{cfg.get('missingness',{}).get('mechanism','?')}_{cfg.get('missingness',{}).get('pattern','general')}_{rate_pct}pct_s{seed}"
            if not (Path("results") / "raw" / f"{eid}.json").exists():
                test_cfg = cfg
                break

    if test_cfg:
        ds = test_cfg.get("dataset",{}).get("name","?")
        method = test_cfg.get("method",{}).get("name","?")
        print(f"Testing with: {ds}/{method}")
        
        ctx = mp.get_context("spawn")
        t0 = time.time()
        with ProcessPoolExecutor(
            max_workers=1, mp_context=ctx,
            initializer=partial(_worker_init, 1),
        ) as executor:
            future = executor.submit(_worker_run_experiment, test_cfg)
            try:
                result = future.result(timeout=120)
                print(f"  Result: {result}")
            except Exception as e:
                print(f"  Error: {type(e).__name__}: {e}")
        print(f"  Time: {time.time()-t0:.2f}s")
    else:
        print("All experiments already done!")

    # Test with 4 workers, 20 items
    print("\n=== Test: 4 workers, 20 pending items ===")
    test_cfgs = []
    for cfg in configs:
        ds = cfg.get("dataset",{}).get("name","?")
        method = cfg.get("method",{}).get("name","?")
        rate_pct = int(round(cfg.get("missingness",{}).get("rate",0)*100))
        seed = cfg.get("project",{}).get("seed",0)
        mechanism = cfg.get("missingness",{}).get("mechanism","?")
        pattern = cfg.get("missingness",{}).get("pattern","general")
        eid = f"{ds}_{method}_{mechanism}_{pattern}_{rate_pct}pct_s{seed}"
        if not (Path("results") / "raw" / f"{eid}.json").exists():
            test_cfgs.append(cfg)
            if len(test_cfgs) >= 20:
                break
    
    if test_cfgs:
        print(f"Running {len(test_cfgs)} experiments...")
        t0 = time.time()
        with ProcessPoolExecutor(
            max_workers=4, mp_context=ctx,
            initializer=partial(_worker_init, 4),
        ) as executor:
            futures = {executor.submit(_worker_run_experiment, c): i for i, c in enumerate(test_cfgs)}
            done = 0
            for f in as_completed(futures):
                try:
                    r = f.result(timeout=300)
                    done += 1
                    status = r.get("status", "?")
                    tag = r.get("tag", "?")
                    print(f"  [{done}/{len(test_cfgs)}] {status}: {tag} ({time.time()-t0:.1f}s)")
                except Exception as e:
                    done += 1
                    print(f"  [{done}/{len(test_cfgs)}] ERROR: {e}")
        print(f"Complete in {time.time()-t0:.2f}s")
