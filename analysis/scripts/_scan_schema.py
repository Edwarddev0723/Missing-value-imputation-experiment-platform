"""Scan result JSONs to understand schema, field coverage, and variations."""
import json
import glob
import collections
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "results", "raw")

files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
print(f"Total result files found: {len(files)}")

task_types = collections.Counter()
ds_tasks = {}
has_empty_metrics = 0
has_null_fields = collections.Counter()
all_keys = collections.Counter()
downstream_keys = collections.Counter()
datasets = set()
methods = set()
mechanisms = set()
patterns = set()
rates = set()
seeds = set()

for f in files:
    d = json.load(open(f))
    for k in d:
        all_keys[k] += 1
    
    tt = d.get("downstream_performance", {}).get("task_type", "?")
    task_types[tt] += 1
    ds = d["dataset_name"]
    ds_tasks[ds] = tt
    datasets.add(ds)
    methods.add(d["method_name"])
    mechanisms.add(d["mechanism"])
    patterns.add(d["pattern"])
    rates.add(d["missing_rate"])
    seeds.add(d["seed"])
    
    dm = d.get("downstream_performance", {}).get("metrics", {})
    if not dm:
        has_empty_metrics += 1
    for mk in dm:
        downstream_keys[mk] += 1
    
    iq = d.get("imputation_quality", {})
    num = iq.get("numerical", {})
    cat = iq.get("categorical", {})
    if cat.get("accuracy") is None:
        has_null_fields["cat_accuracy_null"] += 1
    if num.get("rmse") is None:
        has_null_fields["num_rmse_null"] += 1
    dp = d.get("distribution_preservation", {})
    if dp.get("jsd_mean") is None:
        has_null_fields["jsd_null"] += 1
    if dp.get("wasserstein_distance_mean") is None:
        has_null_fields["wasserstein_null"] += 1

print(f"\nDatasets ({len(datasets)}): {sorted(datasets)}")
print(f"Methods ({len(methods)}): {sorted(methods)}")
print(f"Mechanisms ({len(mechanisms)}): {sorted(mechanisms)}")
print(f"Patterns ({len(patterns)}): {sorted(patterns)}")
print(f"Rates ({len(rates)}): {sorted(rates)}")
print(f"Seeds ({len(seeds)}): {sorted(seeds)}")

print(f"\nTask types: {dict(task_types)}")
print(f"Dataset -> task type: {ds_tasks}")

print(f"\nTop-level keys (count out of {len(files)}):")
for k, v in sorted(all_keys.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\nDownstream metric keys (count out of {len(files)}):")
for k, v in sorted(downstream_keys.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

print(f"\nEmpty downstream metrics: {has_empty_metrics}")
print(f"\nNull field counts:")
for k, v in sorted(has_null_fields.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} / {len(files)} ({100*v/len(files):.1f}%)")
