from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from dataset import clean, label_encode, FashionDataset, transformations
from sklearn.model_selection import train_test_split

# load precomputed embeddings
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

# query: find a tshirt and search top 5 similar
tshirt_indices = (label_array == class_names.index("Tshirts")).nonzero(as_tuple=True)[0]
tshirt_embedding = embedding_matrix[tshirt_indices[0].item()]
tshirt_normalized = F.normalize(tshirt_embedding, p=2, dim=0)
print(f"Norm of normalized T-shirt embedding: {torch.linalg.norm(tshirt_normalized).item()} and difference from 1: {torch.abs(torch.linalg.norm(tshirt_normalized) - 1.0).item()}")

distances, indices = index.search(tshirt_normalized.unsqueeze(0).numpy(), k=6)
for i in range(6):
    img_id = id_list[int(indices[0][i])]
    img = Image.open(f"data/images/{img_id}.jpg")
    img.show()
    if i>0: print(f"result{i} = image ID {img_id} (distance: {distances[0][i]:.4f})")
    else: print(f"query image: {img_id}")


# save the index
faiss.write_index(index, "data/index_baseline.index")
print("index saved to data/index_baseline.index")
