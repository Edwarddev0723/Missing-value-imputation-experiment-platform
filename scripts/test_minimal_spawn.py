"""Absolute minimal spawn test to isolate the hang."""
import sys, os
sys.path.insert(0, ".")

def simple_worker(x):
    """Trivial worker — no imports."""
    return x * 2

def sklearn_worker(x):
    """Worker that imports sklearn."""
    import warnings
    warnings.simplefilter("ignore")
    from sklearn.ensemble import RandomForestClassifier
    return {"result": x * 2, "sklearn": "ok"}

def full_worker(cfg):
    """Worker that does what _worker_run_experiment does."""
    import warnings
    warnings.simplefilter("ignore")
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
    
    from src.factory import ComponentFactory
    from src.orchestration.runner import ExperimentRunner
    
    return {"status": "ok", "imported": True}

if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    ctx = mp.get_context("spawn")

    # Test 1: trivial worker
    print("Test 1: trivial worker...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        futs = [ex.submit(simple_worker, i) for i in range(4)]
        for f in as_completed(futs):
            print(f"  {f.result()}", end=" ")
    print(f"\n  OK ({time.time()-t0:.1f}s)")

    # Test 2: sklearn import worker
    print("\nTest 2: sklearn import worker...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        futs = [ex.submit(sklearn_worker, i) for i in range(4)]
        for f in as_completed(futs):
            print(f"  {f.result()}", end=" ")
    print(f"\n  OK ({time.time()-t0:.1f}s)")

    # Test 3: full import worker
    print("\nTest 3: full factory import worker...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        futs = [ex.submit(full_worker, {"test": i}) for i in range(4)]
        for f in as_completed(futs):
            try:
                r = f.result(timeout=60)
                print(f"  {r}", end=" ")
            except Exception as e:
                print(f"  ERROR: {e}", end=" ")
    print(f"\n  OK ({time.time()-t0:.1f}s)")

    print("\nAll tests passed!")
