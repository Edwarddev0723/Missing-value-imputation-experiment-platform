"""Minimal test to reproduce parallel hang."""
import sys
sys.path.insert(0, ".")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time

# Test 1: Can _worker_init + partial pickle correctly with spawn?
def worker_init(max_workers):
    import warnings
    warnings.simplefilter("ignore")
    import os
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

def worker_fn(cfg):
    import warnings
    warnings.simplefilter("ignore")
    return {"status": "ok", "tag": cfg.get("tag", "?")}

print("Test 1: spawn + partial initializer + small batch")
ctx = mp.get_context("spawn")
t0 = time.time()
with ProcessPoolExecutor(
    max_workers=2, mp_context=ctx,
    initializer=partial(worker_init, 2),
) as executor:
    futures = {executor.submit(worker_fn, {"tag": f"test_{i}"}): i for i in range(10)}
    for f in as_completed(futures):
        r = f.result()
print(f"  OK in {time.time()-t0:.2f}s")

# Test 2: Larger batch - 1000 items 
print("\nTest 2: spawn + 1000 items")
t0 = time.time()
with ProcessPoolExecutor(
    max_workers=4, mp_context=ctx,
    initializer=partial(worker_init, 4),
) as executor:
    futures = {executor.submit(worker_fn, {"tag": f"test_{i}"}): i for i in range(1000)}
    done = 0
    for f in as_completed(futures):
        r = f.result()
        done += 1
        if done % 250 == 0:
            print(f"  {done}/1000 in {time.time()-t0:.2f}s")
print(f"  OK in {time.time()-t0:.2f}s")

# Test 3: Actual experiment config with real factory import
print("\nTest 3: Real experiment config with factory import")
from src.config import load_yaml, expand_batch_config
batch_config = load_yaml("configs/experiments/full_benchmark_nondl.yaml")
experiment_configs = expand_batch_config(batch_config, "configs")
print(f"  Total configs: {len(experiment_configs)}")

# Test single worker function with real config  
cfg = experiment_configs[0]
print(f"  Config keys: {sorted(cfg.keys())}")
print(f"  method.params: {cfg.get('method', {}).get('params', 'MISSING')}")

# Run the real worker on a few configs
from src.orchestration.batch_runner import _worker_run_experiment, _worker_init

print("\nTest 4: Real worker, 3 configs, 2 workers")
t0 = time.time()
# Pick 3 non-skippable configs (use different seeds to find ones not done)
test_cfgs = experiment_configs[:3]
with ProcessPoolExecutor(
    max_workers=2, mp_context=ctx,
    initializer=partial(_worker_init, 2),
) as executor:
    futures = {executor.submit(_worker_run_experiment, c): i for i, c in enumerate(test_cfgs)}
    for f in as_completed(futures):
        try:
            r = f.result(timeout=120)
            print(f"  Result: {r}")
        except Exception as e:
            print(f"  Error: {e}")
print(f"  Done in {time.time()-t0:.2f}s")

print("\nAll tests passed!")
