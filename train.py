import os
import numpy as np
import pandas as pd
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import pandas as pd

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import timm

import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_path = "train_thumbnails/"
csv_path = "train.csv"




class UBCDataset(Dataset):
    def __init__(self, csv_path, imgs_path, transforms=None):
        self.csv_path = csv_path
        self.imgs_path = imgs_path
        self.transform = transforms
        self.df = pd.read_csv(csv_path)

        self.imgs = []
        self.labels = []

        for idx, row in self.df.iterrows():
            img_id = row['image_id']
            is_tma = row['is_tma']
            label = row["label"]
            img_file_path = os.path.join(self.imgs_path, str(img_id) + '_thumbnail.png')
            if os.path.isfile(img_file_path):
                img = Image.open(img_file_path).convert('RGB')
                # img = cv2.imread(img_file_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)

                self.imgs.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


transforms = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ]
)

ubc_dataset = UBCDataset(csv_path, img_path, transforms)

ubc_dataloader = DataLoader(
    dataset=ubc_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

for data in ubc_dataloader:
    print(data)





class UBCModel_ResNet50(nn.Module):
    def __init__(self, img_size):
        super(UBCModel_ResNet50, self).__init__()
        self.num_class = 5
        self.model = timm.create_model(
            model_name="resnet50",
            pretrained=None,
            in_chans = 3,
        )

        self.out_feature = self.model.fc.out_features
        self.fc1 = nn.Linear(self.out_feature, self.out_feature // 2)
        self.fc2 = nn.Linear(self.out_feature // 2, self.num_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

df = pd.read_csv(csv_path)
imgs = []
labels = []
dataset_dist = {}

for idx, row in df.iterrows():
    img_id = row['image_id']
    is_tma = row['is_tma']
    label  = row["label"]
    img_file_path = os.path.join(img_path, str(img_id) + '_thumbnail.png')
    if os.path.isfile(img_file_path) and is_tma == 'FALSE':
        img = Image.open(img_file_path).convert('RGB')
        img = torchvision.transforms.ToTensor(img)
        imgs.append(img)
        labels.append(label)
        cnt = cnt + 1

print(cnt)

train_dataset = {
    'train_images': imgs,
    'labels': labels
}

def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None):
    return confusion_matrix(y_true, y_pred, labels, sample_weight)

def disp_confusion_matrix(confusion_mat, display_labels):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=display_labels)
    disp.plot(
        include_values=True,
        cmap='viridis',
        ax=None,
        xtricks_rotation="horizontal",
        values_format='d'
    )



