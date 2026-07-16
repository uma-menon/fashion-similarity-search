import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from src.dataset import clean, label_encode, FashionDataset, transformations
from src.embeddings import embed_dataset, cluster_sanity_check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = clean()
class_names, class_to_idx = label_encode(df)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)

train_dataset = FashionDataset(train_df, class_to_idx, transformations)
val_dataset = FashionDataset(val_df, class_to_idx, transformations)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(2048, len(class_names))  # Adjust the final layer for the number of classes

#freeze backbone
for param in model.parameters(): param.requires_grad = False
#unfreeze fc layer
for param in model.fc.parameters(): param.requires_grad = True

expected_trainable_params = 2049 * len(class_names)
print(f"Expect {expected_trainable_params} trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_trainable_params}")

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Adam automatically adjusts learning rate for each parameter


def training_loop(num_epochs):
    best_val_acc = 0.0

    for epoch in range(num_epochs):

        # --- train phase ---
        model.train()
        train_batch_losses = []
        train_correct, train_total = 0, 0

        for batch_idx, (images, labels, ids) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # track running loss:
            train_batch_losses.append(loss.item())

            # track # of correct predictions:
            predictions = torch.argmax(outputs, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            if batch_idx % 50 == 0:
                print(f"  epoch {epoch+1} | batch {batch_idx}/{len(train_loader)} | loss: {loss.item():.4f}")



        # --- val phase ---
        model.eval()
        val_batch_losses = []
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels, ids in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_batch_losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)


        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "data/best_model.pt")
            print(f"  checkpoint saved (val acc: {best_val_acc*100:.2f}%)")

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"train loss: {sum(train_batch_losses)/len(train_batch_losses):.4f} | "
            f"train acc: {train_acc*100:.2f}% | "
            f"val loss: {sum(val_batch_losses)/len(val_batch_losses):.4f} | "
            f"val acc: {val_acc*100:.2f}%\n"
        )
#end


if __name__ == '__main__':
    training_loop(num_epochs=5)

    model = resnet50(weights=None)
    model.fc = nn.Linear(2048, len(class_names))
    model.load_state_dict(torch.load('data/best_model.pt', map_location=device))
    model.to(device)
    model.eval()
    model.fc = nn.Identity()
    embedding_matrix, label_array, id_list = embed_dataset(model, val_loader)
    torch.save(embedding_matrix, 'data/embeddings_finetuned.pt')
    torch.save(label_array, 'data/labels_finetuned.pt')
    np.save('data/id_list_finetuned.npy', np.array(id_list))

    cluster_sanity_check(embedding_matrix, label_array, class_names)