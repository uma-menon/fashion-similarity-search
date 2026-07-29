import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

from dataset import clean, label_encode
from benchmark import get_ground_truth, METADATA_CATEGORIES, GROUPING_CATEGORIES


faiss.omp_set_num_threads(1)

USE_FINETUNED = False
USE_FULL = True
INDEX_TYPE = "ivf" # "flat" or "ivf"

PREFIX = ("finetuned" if USE_FINETUNED else "baseline") + ("_full" if USE_FULL else "_val") + ("_ivf" if INDEX_TYPE=="ivf" else "")
print(f"evaluating: {PREFIX}")

# load embeddings
if USE_FULL and USE_FINETUNED:
    embedding_matrix = torch.load('data/embeddings_full_finetuned.pt')
    id_list = np.load('data/id_list_full_finetuned.npy', allow_pickle=False).tolist()
elif USE_FULL:
    embedding_matrix = torch.load('data/embeddings_full.pt')
    id_list = np.load('data/id_list_full.npy', allow_pickle=False).tolist()
elif USE_FINETUNED:
    embedding_matrix = torch.load('data/embeddings_finetuned.pt')
    id_list = np.load('data/id_list_finetuned.npy', allow_pickle=False).tolist()
else:
    embedding_matrix = torch.load('data/embeddings.pt')
    id_list = np.load('data/id_list.npy', allow_pickle=False).tolist()

# normalize once upfront: all queries use the same normalized matrix
normalized_matrix = F.normalize(embedding_matrix, p=2, dim=1).numpy()

# id_list holds ints; building str->int index lookup for benchmark query IDs (= strings)
id_to_index = {str(img_id): i for i, img_id in enumerate(id_list)}

# load index
index_path = f"data/index_{PREFIX}.index"
index = faiss.read_index(index_path)
if INDEX_TYPE == "ivf": index.nprobe = 10
print(f"index loaded: {index.ntotal} vectors from {index_path}")

# load benchmark + df
with open('data/benchmark_queries.json') as f:
    benchmark_queries = json.load(f)

df = clean()
class_names, _ = label_encode(df)


def precision_at_k(result_ids, ground_truth, k):
    """Fraction of top-k results that are in ground_truth."""
    hits = sum(1 for r in result_ids[:k] if r in ground_truth)
    return hits / k


def retrieve(query_id_str, k=10):
    """
    Given a query image ID string, returns up to k result ID strings
    (self-match excluded). Also returns avg query latency in ms.
    """
    if query_id_str not in id_to_index:
        return None, None

    idx = id_to_index[query_id_str]
    query_vec = normalized_matrix[idx].reshape(1, -1)   # already normalized

    t0 = time.perf_counter()
    distances, indices = index.search(query_vec, k + 1)
    latency_ms = (time.perf_counter() - t0) * 1000

    result_ids = [str(id_list[int(indices[0][i])]) for i in range(1, k + 1)]
    return result_ids, latency_ms


# main eval loop
K_VALUES = [1, 3, 5, 10]
results = {} # category -> {p@1, p@3, p@5, p@10, avg_latency_ms}
skipped = 0

all_categories = list(METADATA_CATEGORIES.keys()) + list(GROUPING_CATEGORIES.keys())

for category in all_categories:
    query_ids = benchmark_queries.get(category, [])
    cat_scores = {k: [] for k in K_VALUES}
    cat_latencies = []

    for query_id in query_ids:
        result_ids, latency_ms = retrieve(query_id, k=max(K_VALUES))

        if result_ids is None:
            print(f"  WARNING: {query_id} not in id_list, skipping")
            skipped += 1
            continue

        ground_truth = get_ground_truth(query_id, category, df)

        for k in K_VALUES:
            cat_scores[k].append(precision_at_k(result_ids, ground_truth, k))

        cat_latencies.append(latency_ms)

    results[category] = {
        **{f"p@{k}": round(sum(cat_scores[k]) / len(cat_scores[k]), 4) if cat_scores[k] else 0.0
           for k in K_VALUES},
        "avg_latency_ms": round(sum(cat_latencies) / len(cat_latencies), 3) if cat_latencies else 0.0,
        "n_queries": len(cat_latencies),
    }

if skipped:
    print(f"\nWARNING: {skipped} query IDs not found in id_list")

# print results table
print(f"\n{'category':<22} {'P@1':>6} {'P@3':>6} {'P@5':>6} {'P@10':>6} {'ms':>8}")
print("-" * 58)
for category, scores in results.items():
    print(f"{category:<22} {scores['p@1']:>6.3f} {scores['p@3']:>6.3f} {scores['p@5']:>6.3f} {scores['p@10']:>6.3f} {scores['avg_latency_ms']:>7.2f}ms")

p5_vals = [v['p@5'] for v in results.values()]
print(f"\n{'MEAN':<22} {sum(v['p@1'] for v in results.values())/len(results):>6.3f} "
      f"{sum(v['p@3'] for v in results.values())/len(results):>6.3f} "
      f"{sum(p5_vals)/len(p5_vals):>6.3f} "
      f"{sum(v['p@10'] for v in results.values())/len(results):>6.3f}")

# save results
out_path = f"results/eval_{PREFIX}.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nresults saved to {out_path}")