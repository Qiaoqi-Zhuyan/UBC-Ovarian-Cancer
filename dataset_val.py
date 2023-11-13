import pandas as pd
import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


df = pd.read_csv("train.csv")
img_path = "train_thumbnails/"
csv_path = "train.csv"


cnt = 0

train_data = {}

for idx, row in df.iterrows():
    img_id = row['image_id']
    is_tma = row['is_tma']
    label  = row["label"]
    img_file_path = os.path.join(img_path, str(img_id) + '_thumbnail.png')
    if os.path.isfile(img_file_path):
        train_data.update({f'{img_file_path}': f'{label}'})
        #print(img_file_path)
        cnt = cnt + 1
        #print(is_tma)
#    else:
#        print(img_file_path)
#        print(is_tma)

#print(df.iloc[0]['image_id'])




class UBCDataset(Dataset):
    def __init__(self, csv_path, imgs_path, transforms=None):
        self.csv_path = csv_path
        self.imgs_path = imgs_path
        self.transform = transforms
        self.df = pd.read_csv(csv_path)

        self.imgs = []
        self.labels = []

        for idx, row in df.iterrows():
            img_id = row['image_id']
            is_tma = row['is_tma']
            label = row["label"]
            img_file_path = os.path.join(self.imgs_path, str(img_id) + '_thumbnail.png')
            if os.path.isfile(img_file_path) :
                img = Image.open(img_file_path)
                self.imgs.append(img)
                self.labels.append(label)



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


ubc_dataset = UBCDataset(csv_path , img_path)
print(ubc_dataset.__getitem__(2))





