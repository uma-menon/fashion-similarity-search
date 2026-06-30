import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


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
    # print("valid_articles: ", valid_articles)
    # print(len(valid_articles))
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
    # print(class_names)
    print(f"\nclass_to_idx: {class_to_idx}\n")
    # print("number of classes: ", len(class_names))
    return class_names, class_to_idx


transformations=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], #imageNet mean 
                         std=[0.229, 0.224, 0.225]) #imageNet std
                        #  (pixel - mean) / std
])

class Dataset:

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
        if self.image_transform:
            img=self.image_transform(img)
        return img, label

df=clean()
class_names, class_to_idx = label_encode(df)

print('--- WITHOUT TRANSFORM ---')
dataset=Dataset(df, class_to_idx)
image, label = dataset[0]
print("Image type:", type(image))
print("Label:", label)

print('--- WITH TRANSFORM ---')
dataset=Dataset(df, class_to_idx, transformations)
image, label = dataset[0]
print("Image type:", type(image))
print("Tensor shape:", image.shape)
print("Label:", label)

print('--- DATALOADERS ---')

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)
train_dataset=Dataset(train_df, class_to_idx, transformations)
val_dataset=Dataset(val_df, class_to_idx, transformations)


train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=32, shuffle=False)

print("--- ONE BATCH ---")
check_batch = next(iter(train_loader))
print("Batch image shape:", check_batch[0].shape)
print("Batch label shape and dtype:", check_batch[1].shape, check_batch[1].dtype)
print("Batch label values:", check_batch[1])
range_check=[check_batch[1][i].item() in range(0,39) for i in range(32)]
print(not False in range_check) # should be False if all labels are in range 0-38

print("--- RECONSTRUCTED IMAGES ---")
for i in range(4):
    image_tensor=check_batch[0][i]
    print("Supposed to be: ", check_batch[1][i].item() , class_names[check_batch[1][i].item()])
    mean_tensor= torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_tensor= torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    px = image_tensor * std_tensor + mean_tensor
    px = px * 255
    px = px.permute(1, 2, 0) # CHW to HWC
    img_arr=px.detach().numpy().astype('uint8')
    img=Image.fromarray(img_arr)
    # img.show()
    