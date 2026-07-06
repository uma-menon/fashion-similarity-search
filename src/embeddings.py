import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import clean, label_encode, FashionDataset, transformations


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def embed_dataset(model, loader):
    all_embeddings = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch_idx, (images, labels, ids) in enumerate(loader):
            images = images.to(device)
            embeddings = model(images)          # [batch_size, 2048]
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            all_ids.append(list(ids))
            if batch_idx % 10 == 0: print(f"batch {batch_idx}/{len(loader)}")

    embedding_matrix = torch.cat(all_embeddings, dim=0)    # [num_images, 2048]
    label_array = torch.cat(all_labels, dim=0)              # [num_images]
    id_list = sum(all_ids, [])                              # flat list of strings

    print("embedding matrix shape:", embedding_matrix.shape)
    print("label array shape:", label_array.shape)
    print("id list length:", len(id_list))

    return embedding_matrix, label_array, id_list


def cluster_sanity_check(embedding_matrix, label_array, class_names):

    tshirt_indices = (label_array == class_names.index("Tshirts")).nonzero(as_tuple=True)[0]
    heels_indices = (label_array == class_names.index("Heels")).nonzero(as_tuple=True)[0]

    emb_tshirt_0 = embedding_matrix[tshirt_indices[0].item()]
    emb_tshirt_1 = embedding_matrix[tshirt_indices[1].item()]
    emb_heels_0 = embedding_matrix[heels_indices[0].item()]

    same_class_dist = torch.dist(emb_tshirt_0, emb_tshirt_1).item()
    diff_class_dist = torch.dist(emb_tshirt_0, emb_heels_0).item()

    print(f"\n--- cluster sanity check ---")
    print(f"same-class L2 (Tshirts vs Tshirts): {same_class_dist:.4f}")
    print(f"diff-class L2 (Tshirts vs Heels): {diff_class_dist:.4f}")
    print(f"same < diff: {same_class_dist < diff_class_dist}")


if __name__ == '__main__':
    df = clean()
    class_names, class_to_idx = label_encode(df)

    _, val_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)

    val_dataset = FashionDataset(val_df, class_to_idx, transformations)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = build_model()

    # single-image sanity check
    images, labels, ids = next(iter(val_loader))
    with torch.no_grad():
        output = model(images[0].unsqueeze(0).to(device))
    print("single image output shape:", output.shape)  # expect [1, 2048]

    # embed full val set
    embedding_matrix, label_array, id_list = embed_dataset(model, val_loader)

    cluster_sanity_check(embedding_matrix, label_array, class_names)

    torch.save(embedding_matrix, 'data/embeddings.pt')
    torch.save(label_array, 'data/labels.pt')
    np.save('data/id_list.npy', np.array(id_list))
