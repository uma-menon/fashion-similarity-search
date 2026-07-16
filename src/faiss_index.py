import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from src.dataset import clean, label_encode, FashionDataset, transformations
from sklearn.model_selection import train_test_split

USE_FINETUNED=False
PREFIX = "finetuned" if USE_FINETUNED else "baseline"
print(f"using: {PREFIX} embeddings")

# load precomputed embeddings
if USE_FINETUNED:
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
row_norms = torch.linalg.norm(normalized_embeddings, dim=1)
print(f"Row norms should be close to 1.0: {row_norms}")

# build FAISS index
print(normalized_embeddings.shape[1], "should be 2048")
DIMENSIONALITY= normalized_embeddings.shape[1]
index = faiss.IndexFlatL2(DIMENSIONALITY)  # L2 distance
index.add(normalized_embeddings.numpy())
print(f"Number of vectors in the index: {index.ntotal} should match the number of images in the validation set: {len(val_dataset)}")





def query(faiss_index, embedding_matrix, id_list, label_array, class_names, query_index, k, prefix=""):
    query_id = id_list[query_index]
    query_label = class_names[label_array[query_index].item()]

    savedir = f'notebooks/checks/{prefix}/{query_label}_{query_id}'

    if os.path.exists(savedir): shutil.rmtree(savedir)
    os.makedirs(savedir)

    Image.open(f"data/images/{query_id}.jpg").save(f"{savedir}/query_{query_id}.jpg")
    print(f"query image: {query_id} ({query_label})")

    query_embedding = embedding_matrix[query_index]
    query_normalized = F.normalize(query_embedding, p=2, dim=0)
    distances, indices = faiss_index.search(query_normalized.unsqueeze(0).numpy(), k+1)

    for i in range(1, k+1):
        img_id = id_list[int(indices[0][i])]
        result_label = class_names[label_array[int(indices[0][i])].item()]

        Image.open(f"data/images/{img_id}.jpg").save(f"{savedir}/result_{i}_{img_id}.jpg")
        print(f"  result {i}: {img_id} ({result_label}) | dist: {distances[0][i]:.4f}")


QUERY_IMAGES=["Tshirts", "Heels", "Jeans", "Sunglasses", "Dresses"]
for query_class in QUERY_IMAGES:
    class_indices = (label_array == class_names.index(query_class)).nonzero(as_tuple=True)[0]
    query_index = class_indices[0].item()
    query(index, embedding_matrix, id_list, label_array, class_names, query_index, k=5, prefix=PREFIX)


# save the index
if USE_FINETUNED:
    faiss.write_index(index, "data/index_finetuned.index")
    print("index saved to data/index_finetuned.index")
else:
    faiss.write_index(index, "data/index_baseline.index")
    print("index saved to data/index_baseline.index")