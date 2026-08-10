import json

baseline   = json.load(open('results/eval_baseline_full.json'))
finetuned  = json.load(open('results/eval_finetuned_full.json'))

categories = list(baseline.keys())

print(f"\n{'category':<22} {'Base P@5':>9} {'FT P@5':>9} {'Delta':>8}")
print("-" * 52)
for cat in categories:
    b = baseline[cat]['p@5']
    f = finetuned[cat]['p@5']
    delta = f - b
    sign = "+" if delta >= 0 else ""
    print(f"{cat:<22} {b:>9.3f} {f:>9.3f} {sign}{delta:>7.3f}")

base_mean = sum(baseline[c]['p@5'] for c in categories) / len(categories)
ft_mean   = sum(finetuned[c]['p@5'] for c in categories) / len(categories)
delta_mean = ft_mean - base_mean
sign = "+" if delta_mean >= 0 else ""
print("-" * 52)
print(f"{'MEAN':<22} {base_mean:>9.3f} {ft_mean:>9.3f} {sign}{delta_mean:>7.3f}")

print(f"\n--- latency (flat index) ---")
base_lat = sum(baseline[c]['avg_latency_ms'] for c in categories) / len(categories)
ft_lat   = sum(finetuned[c]['avg_latency_ms'] for c in categories) / len(categories)
print(f"baseline:   {base_lat:.2f}ms avg")
print(f"finetuned:  {ft_lat:.2f}ms avg")

# IVF comparison if available
try:
    ivf = json.load(open('results/eval_baseline_full_ivf.json'))
    ivf_lat = sum(ivf[c]['avg_latency_ms'] for c in categories) / len(categories)
    ivf_mean = sum(ivf[c]['p@5'] for c in categories) / len(categories)
    print(f"\n--- flat vs IVF (baseline embeddings) ---")
    print(f"flat:  P@5={base_mean:.3f} | {base_lat:.2f}ms")
    print(f"IVF:   P@5={ivf_mean:.3f} | {ivf_lat:.2f}ms | speedup: {base_lat/ivf_lat:.1f}x | precision loss: {base_mean-ivf_mean:.3f}")
except FileNotFoundError:
    pass