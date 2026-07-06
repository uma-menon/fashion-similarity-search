import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


def clean():
        
    df = pd.read_csv('data/styles.csv', on_bad_lines='skip')
    print("before clean:", df.shape)

    # DROP rows with missing images
    mask = df['id'].apply(lambda x: os.path.exists(f"data/images/{x}.jpg"))
    df = df[mask].reset_index(drop=True)
    # print(df.shape)

    # DROP underrepresented articles
    THRESHOLD = 150
    val_counts=df['articleType'].value_counts()
    valid_articles=val_counts[val_counts>THRESHOLD].index.tolist()
    df = df[df['articleType'].isin(valid_articles)].reset_index(drop=True)
    # print(df.shape)

    # DROP non-fashion articles
    non_fashion_articles = {"Wallets", "Backpacks", "Perfume and Body Mist", "Deodorant", "Nail Polish", "Lipstick"}
    df = df[~df['articleType'].isin(non_fashion_articles)].reset_index(drop=True)
    print("after clean:", df.shape)

    return df


def label_encode(df):
    class_names = sorted(df['articleType'].unique().tolist())
    class_to_idx = {v: i for i, v in enumerate(class_names)}
    # print(f"\nclass_to_idx: {class_to_idx}\n")
    print(f"number of classes: {len(class_names)}")
    return class_names, class_to_idx


transformations=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #imageNet mean 
                         std=[0.229, 0.224, 0.225]) #imageNet std
                        #  (pixel - mean) / std
])

class FashionDataset(Dataset):

    def __init__(self, df, class_to_idx, image_transform=None):
        self.df = df
        self.class_to_idx = class_to_idx
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        img_path=f"data/images/{row['id']}.jpg"
        img=Image.open(img_path).convert('RGB')
        label=self.class_to_idx[row['articleType']]
        if self.image_transform: img=self.image_transform(img)
        return img, label, row['id']



if __name__ == '__main__':
    df = clean()
    class_names, class_to_idx = label_encode(df)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['articleType'], random_state=42
    )

    train_dataset = FashionDataset(train_df, class_to_idx, transformations)
    val_dataset = FashionDataset(val_df, class_to_idx, transformations)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"train: {len(train_dataset)} | val: {len(val_dataset)}")

    images, labels, ids = next(iter(train_loader))
    print("batch image shape:", images.shape)
    print("batch label shape:", labels.shape, "| dtype:", labels.dtype)