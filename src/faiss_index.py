import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import json
from sklearn.model_selection import train_test_split

from dataset import clean, label_encode, FashionDataset, transformations
from benchmark import get_ground_truth


faiss.omp_set_num_threads(1)

USE_FINETUNED=False
USE_FULL=True

PREFIX = ("finetuned" if USE_FINETUNED else "baseline") + ("_full" if USE_FULL else "_val")
print(f"using: {PREFIX} embeddings")



# load precomputed embeddings
if USE_FULL and USE_FINETUNED:
    embedding_matrix = torch.load('data/embeddings_full_finetuned.pt')
    label_array = torch.load('data/labels_full_finetuned.pt')
    id_list = np.load('data/id_list_full_finetuned.npy', allow_pickle=False).tolist()
elif USE_FULL:
    embedding_matrix = torch.load('data/embeddings_full.pt')
    label_array = torch.load('data/labels_full.pt')
    id_list = np.load('data/id_list_full.npy', allow_pickle=False).tolist()
elif USE_FINETUNED:
    embedding_matrix = torch.load('data/embeddings_finetuned.pt')
    label_array = torch.load('data/labels_finetuned.pt')
    id_list = np.load('data/id_list_finetuned.npy', allow_pickle=False).tolist()
else:
    embedding_matrix = torch.load('data/embeddings.pt')
    label_array = torch.load('data/labels.pt')
    id_list = np.load('data/id_list.npy', allow_pickle=False).tolist()


# rebuild class_names and val_dataset
df = clean()
class_names, class_to_idx = label_encode(df)
_, val_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)
val_dataset = FashionDataset(val_df, class_to_idx, transformations)

# normalize embedding matrix
normalized_embeddings = F.normalize(embedding_matrix, p=2, dim=1)
print(f"row norms (first 5): {torch.linalg.norm(normalized_embeddings, dim=1)[:5]}")
print(f"dimensionality: {normalized_embeddings.shape[1]}  (expect 2048)")

# build FAISS index
index = faiss.IndexFlatL2(normalized_embeddings.shape[1])
index.add(normalized_embeddings.numpy())
print(f"index size: {index.ntotal}  |  embeddings: {len(id_list)}")





def query(faiss_index, embedding_matrix, id_list, label_array, class_names, query_index, k, prefix=""):
    query_id = id_list[query_index]
    query_label = class_names[label_array[query_index].item()]
    savedir = f'notebooks/checks/{prefix}/{query_label}_{query_id}'

    if os.path.exists(savedir): shutil.rmtree(savedir)
    os.makedirs(savedir)

    Image.open(f"data/images/{query_id}.jpg").save(f"{savedir}/query.jpg")
    print(f"query image: {query_id} ({query_label})")

    query_embedding = embedding_matrix[query_index]
    query_normalized = F.normalize(query_embedding, p=2, dim=0)
    distances, indices = faiss_index.search(query_normalized.unsqueeze(0).numpy(), k+1)

    for i in range(1, k+1):
        img_id = id_list[int(indices[0][i])]
        result_label = class_names[label_array[int(indices[0][i])].item()]

        Image.open(f"data/images/{img_id}.jpg").save(f"{savedir}/result_{i}_{img_id}.jpg")
        print(f"  result {i}: {img_id} ({result_label}) | dist: {distances[0][i]:.4f}")


QUERY_CLASSES=["Tshirts", "Heels", "Jeans", "Sunglasses", "Dresses"]
for query_class in QUERY_CLASSES:
    class_indices = (label_array == class_names.index(query_class)).nonzero(as_tuple=True)[0]
    query_index = class_indices[0].item()
    query(index, embedding_matrix, id_list, label_array, class_names, query_index, k=5, prefix=PREFIX)


# save the index
index_path = f"data/index_{PREFIX}.index"
faiss.write_index(index, index_path)
print(f"index saved to {index_path}")


# --- end-to-end benchmark preview ---

with open('data/benchmark_queries.json') as f:
    benchmark_queries = json.load(f)

# pick one category and one query to test
preview_category = "footwear"
preview_query_id = int(benchmark_queries[preview_category][0])

# find its index in id_list
preview_index = id_list.index(preview_query_id)

print(f"\n--- end-to-end preview: {preview_category} ---")
print(f"query id: {preview_query_id}")

# run query
query(index, embedding_matrix, id_list, label_array, class_names, preview_index, k=5, prefix=f"preview_{PREFIX}")

# check how many results are in ground truth
from benchmark import get_ground_truth
gt = get_ground_truth(preview_query_id, preview_category, df)
distances, indices = index.search(F.normalize(embedding_matrix[preview_index], p=2, dim=0).unsqueeze(0).numpy(), 6)
result_ids = [id_list[int(indices[0][i])] for i in range(1, 6)]
hits = sum(1 for r in result_ids if str(r) in gt)
print(f"precision@5: {hits}/5 = {hits/5:.2f}")