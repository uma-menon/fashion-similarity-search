import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader

from dataset import clean, label_encode, FashionDataset, transformations
from embeddings import embed_dataset, cluster_sanity_check


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = clean()
class_names, class_to_idx = label_encode(df)

full_dataset = FashionDataset(df, class_to_idx, transformations)
full_loader  = DataLoader(full_dataset, batch_size=64, shuffle=False, num_workers=0)

# load full fine-tuned checkpoint, strip classifier head
model = resnet50(weights=None)
model.fc = nn.Linear(2048, len(class_names))
model.load_state_dict(torch.load('data/best_model_fulltune.pt', map_location=device))
model.fc = nn.Identity()
model.eval()
model.to(device)

# single-image check
images, labels, ids = next(iter(full_loader))
with torch.no_grad():
    out = model(images[0].unsqueeze(0).to(device))
print("single image output shape:", out.shape)  # expect [1, 2048]

# embed full dataset
embedding_matrix, label_array, id_list = embed_dataset(model, full_loader)

torch.save(embedding_matrix, 'data/embeddings_full_finetuned.pt')
torch.save(label_array,      'data/labels_full_finetuned.pt')
np.save('data/id_list_full_finetuned.npy', np.array(id_list))

cluster_sanity_check(embedding_matrix, label_array, class_names)