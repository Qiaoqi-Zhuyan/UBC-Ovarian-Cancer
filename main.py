import os
import random

import numpy as np
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import timm

import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

import logging
import datetime

from tqdm import tqdm
from transformer.mobilevit import mobilevit_xs, mobilevit_xxs, mobilevit_s
#from cnn.convNext import convnext_small, convnext_tiny
from cnn.convNext_official import convnext_tiny, UBC_ConvNext_tiny
import warnings




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
img_path = "/root/autodl-tmp/train_thumbnails"
csv_path = "/root/autodl-tmp/train.csv"

use_aug_dataset = True
aug_img_path = {
    "origin-imgs": "/root/autodl-tmp/train_thumbnails",
    "augs-imgs": "/root/autodl-tmp/train_thumbnails_augs/train_thumbnails_augs"
}

use_init_weights = False
weights_path = "/root/autodl-tmp/ovarian-caner-ubc/mobilevit_s-38a5a959.pth"

epoch_num = 50
timm_model_name = "mobilevit_s"
version = 2
batch_size = 16
lr = 1e-4
img_size = (224, 224)
use_xavier_init = False
weight_decay = 1e-4

model = convnext_tiny(num_classes=5, pretrained=False)
#logger_name = f"{timm_model_name}-{datetime.date.today()}-ver{version}.log"
#save_model_name = f"{timm_model_name}-{datetime.date.today()}-epoch{epoch_num}-init-ver3.pt"
backbone = "convnext_tiny"
save_model_name = f"{backbone}--{datetime.date.today()}--epoch{epoch_num}-ver{version}.pt"
logger_name = f"{backbone}--{datetime.date.today()}--epoch{epoch_num}-ver{version}.log"

warnings.filterwarnings("ignore")

# label transform
label_str2int = {
    'HGSC': 0,
    'EC': 1,
    'CC': 2,
    'LGSC': 3,
    'MC': 4
}


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)


# dataset
class UBCDataset(Dataset):
    def __init__(self, csv_path, imgs_path, transforms=None):
        self.csv_path = csv_path
        self.imgs_path = imgs_path
        self.transform = transforms
        self.df = pd.read_csv(csv_path)

        self.imgs = []
        self.labels = []

        for idx, row in tqdm(self.df.iterrows()):
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


'''
def read_dataset(csv_path, imgs_path, transforms):
    imgs = []
    labels = []
    df = pd.read_csv(csv_path)
    for idx, row in df.iterrows():
        img_id = row['image_id']
        is_tma = row['is_tma']
        label = row["label"]
        img_file_path = os.path.join(imgs_path, str(img_id) + '_thumbnail.png')
        if os.path.isfile(img_file_path):
            img = Image.open(img_file_path).convert('RGB')
            img = transforms(img)
'''

label_encoder = LabelEncoder()


class UBCDataset_lebels(Dataset):
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
                # label_int = label_str2int[f'{label}']
                # img = cv2.imread(img_file_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)
                    # label = torch.tensor(label)

                self.imgs.append(img)
                self.labels.append(label)

        self.labels = label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


# dataset
class UBCDataset_augs(Dataset):
    def __init__(self, csv_path, augs_imgs_dict, transforms=None):
        self.csv_path = csv_path
        self.origin_imgs_path = augs_imgs_dict["origin-imgs"]
        self.augs_imgs_path = augs_imgs_dict["augs-imgs"]
        self.transform = transforms
        self.df = pd.read_csv(csv_path)
        self.augs_imgs_num = 5

        self.imgs = []
        self.labels = []

        for idx, row in tqdm(self.df.iterrows()):
            img_id = row['image_id']
            is_tma = row['is_tma']
            label = row["label"]
            origin_imgs_file = os.path.join(self.origin_imgs_path, str(img_id) + '_thumbnail.png')
            if os.path.isfile(origin_imgs_file):
                img = Image.open(origin_imgs_file).convert('RGB')
                label_int = label_str2int[f'{label}']
                # img = cv2.imread(img_file_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)

                self.imgs.append(img)
                self.labels.append(label_int)

            for i in range(self.augs_imgs_num):
                for j in range(self.augs_imgs_num):
                    augs_imgs_file = os.path.join(self.augs_imgs_path,
                                                  str(img_id) + "_thumbnail" + f"_trans{j + 1}" + f"_{i + 1}.png")
                    if os.path.isfile(augs_imgs_file):
                        img = Image.open(augs_imgs_file).convert('RGB')
                        label_int = label_str2int[f'{label}']
                        # img = cv2.imread(img_file_path)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if self.transform:
                            img = self.transform(img)

                        self.imgs.append(img)
                        self.labels.append(label_int)

        print("Dataset process finish")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


transforms = transforms.Compose(
    [
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

    ]
)

if use_aug_dataset:
    ubc_dataset = UBCDataset_augs(csv_path, aug_img_path, transforms)
else:
    ubc_dataset = UBCDataset(csv_path , img_path, transforms)

train_len = int(len(ubc_dataset) * 0.8)
val_len = len(ubc_dataset) - train_len

train_dataset, val_dataset = torch.utils.data.random_split(
    ubc_dataset, [train_len, val_len]
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
               '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
               ', ' + 'eps=' + str(self.eps) + ')'


# model create
class UBCModel(nn.Module):
    def __init__(self, model_name):
        super(UBCModel, self).__init__()
        self.num_class = 5
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=False,
            num_classes=5
        )
        # in_features = self.model.classifier.in_features
        # self.model.classifier = nn.Identity()
        # self.model.global_pool = nn.Identity()
        # self.pooling = GeM()
        # self.classifier = nn.Linear(in_features, self.num_class)

    def forward(self, x):
        x = self.model(x)
        # x = self.pooling(x).flatten(1)
        # x = self.classifier(x)

        return x


def init_weights_xavier(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)
        if model.bias is not None:
            nn.init.zeros_(model.bias)


class UBCModel_init(nn.Module):
    def __init__(self, model_name, weights_path):
        super(UBCModel_init, self).__init__()
        self.model_ = timm.create_model(
            model_name=model_name,
            pretrained=False
        )
        self.model_.load_state_dict(torch.load(weights_path))
        self.pooling = GeM()
        self.classifier = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.model_(x)
        x = self.pooling(x)
        return self.classifier(x)


# 训练

#if use_init_weights:
#    model = UBCModel_init(timm_model_name, weights_path)
#else:
#    model = UBCModel(timm_model_name)



if use_xavier_init:
    model.apply(init_weights_xavier)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        ft = self.alpha * (1 - pt) ** self.gamma
        focal_loss = ft * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


model = model.to('cuda')
# model.load_state_dict(torch.load(init_weigth))

loss_fn = nn.CrossEntropyLoss()

# moblienet_s 100epochs lr=0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

num_epochs = epoch_num

print(f'device {device}')
print(f'train launch')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(logger_name)
    ]
)

logger = logging.getLogger('training_logger')

logger.info(f'backbone: {backbone}, '
            f'logger: {logger_name}, '
            f'batch_size: {batch_size}, '
            f'epochs: {epoch_num},'
            f'lr: {lr},'
            f'weghts: {save_model_name},'
            f'use_xavier_init: {use_xavier_init},'
            f'transforms: {transforms},'
            f'use_init_weights {use_init_weights}，'
            f'weight_decay {weight_decay}'
            f'use augs_datase: {use_aug_dataset}'
            )
logging.basicConfig(stream=None)
'''
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
                #_, pred = torch.max(y_p, 1)
                pred = torch.argmax(y_p, dim=1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                y = y.to('cpu')
                pred = pred.to('cpu')
                balanced_accuracy = balanced_accuracy_score(y, pred)
            if batch % 10 == 0:
                print(f'Epoch: {epoch}, Accuracy: {100 * correct / total}%, balance accuracy: {balanced_accuracy * 100}%')
                logger.info(f'Epoch: {epoch}, Accuracy: {100 * correct / total}%, balance accuracy: {balanced_accuracy * 100}%')
'''


def train():
    for epoch in range(num_epochs):

        train_correct = 0
        train_total = 0
        train_loss_sum = 0

        model.train()
        with tqdm(train_dataloader, desc='Train') as t:
            for x, y in t:
                t.set_description(f"Epoch [{epoch} / {epoch_num}]")
                x = x.to('cuda')
                y = y.to('cuda')

                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    y_ = torch.argmax(y_pred, dim=1)
                    train_correct += (y_ == y).sum().item()
                    train_total += y.size(0)
                    train_loss_sum += loss.item()
                    training_loss = train_loss_sum / train_total
                    training_acc = train_correct / train_total

                    y = y.to('cpu')
                    y_ = y_.to('cpu')
                    training_balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, y_))

                t.set_postfix(loss=training_loss, training_acc=training_acc, training_BA=training_balanced_accuracy)
                logger.info(f'Train:'
                            f'Epoch: [{epoch} / {epoch_num}, '
                            f'Training Loss: {training_loss}, '
                            f'Training Accuracy: {training_acc},'
                            f'Training Balanced Accuracy: {training_balanced_accuracy}')
                torch.save(model, save_model_name)

            val_correct = 0
            val_total = 0

            model.eval()
            with torch.no_grad():
                with tqdm(val_dataloader, desc="Val") as t2:
                    for x, y in t2:
                        x = x.to('cuda')
                        y = y.to('cuda')
                        y_p = model(x)
                        # _, pred = torch.max(y_p, 1)
                        pred = torch.argmax(y_p, dim=1)

                        val_total += y.size(0)
                        val_correct += (pred == y).sum().item()
                        val_acc = '{:.4f}'.format(val_correct / val_total)

                        y = y.to('cpu')
                        pred = pred.to('cpu')
                        balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, pred))

                        t2.set_postfix(val_acc=val_acc, val_BA=balanced_accuracy)

                        logger.info(f'Val:'
                                    f'Epoch: [{epoch} / {epoch_num}, '
                                    f'Val Accuracy: {val_acc},'
                                    f'val Balanced Accuracy: {balanced_accuracy}')


if __name__ == "__main__":
    train()
