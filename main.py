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

import logging


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_path = "/root/autodl-tmp/train_thumbnails"
csv_path = "/root/autodl-tmp/train.csv"
timm_model_name = "fastvit_s12"
logger_name = "training17.log"
save_model_name = "fastvit_s12-10-30-epoch100.pt"
batch_size = 16
epoch_num = 100
lr = 1e-6

label_str2int = {
    'HGSC': 0,
    'EC': 1,
    'CC': 2,
    'LGSC': 3,
    'MC': 4
}


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
                label_int = label_str2int[f'{label}']
                # img = cv2.imread(img_file_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)

                self.imgs.append(img)
                self.labels.append(label_int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.8721593659261734, 0.7799686061900686, 0.8644588534918227],
            std=[0.08258995918115268, 0.10991684444009092, 0.06839816226731532],
        )
    ]
)

ubc_dataset = UBCDataset(csv_path , img_path, transforms)

train_len = int(len(ubc_dataset) * 0.9)
val_len = len(ubc_dataset) - train_len

train_dataset, val_dataset = torch.utils.data.random_split(
    ubc_dataset, [train_len, val_len]
)


train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)


class UBCModel(nn.Module):
    def __init__(self, model_name):
        super(UBCModel, self).__init__()
        self.num_class = 5
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=None,
            num_classes=5
        )

        #self.out_feature = self.model.fc.out_features
        #self.fc1 = nn.Linear(self.out_feature, self.out_feature // 2)
        #self.fc2 = nn.Linear(self.out_feature // 2, self.num_class)
        #self.linear = nn.Linear(self.out_feature, self.num_class)

    def forward(self, x):
        x = self.model(x)
        #x = self.fc1(x)
        #x = self.fc2(x)
        #x = self.linear(x)

        return x




# 训练
model = UBCModel(timm_model_name)
model = model.to('cuda')
# model.load_state_dict(torch.load(init_weigth))

loss_fn = nn.CrossEntropyLoss()

# moblienet_s 100epochs lr=0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


num_epochs = epoch_num

print(f'device {device}')
print(f'train launch')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(logger_name),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('training_logger')



logger.info(f'backbone: {timm_model_name}, '
            f'logger: {logger_name}, '
            f'batch_size: {batch_size}, '
            f'epochs: {epoch_num},'
            f'lr: {lr},'
            f'weghts: {save_model_name}')

for epoch in range(num_epochs):
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to('cuda')
        y = y.to('cuda')
        #print(y)

        y_pred = model(x)

        #y_pred = torch.max(y_pred, dim=1)
        #print(y_pred)
        # y_pred = torch.max(y_pred, dim=1)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch}, Loss" {loss.item()}')
            logger.info(f'Epoch: {epoch}, Batch: {batch}, Loss" {loss.item()}')
        torch.save(model, save_model_name)

        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for x, y in val_dataloader:
                x = x.to('cuda')
                y = y.to('cuda')
                y_p = model(x)
                _, pred = torch.max(y_p, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
            if batch % 10 == 0:
                print(f'Epoch: {epoch}, Accuracy: {100 * correct / total}%')
                logger.info(f'Epoch: {epoch}, Accuracy: {100 * correct / total}%')

