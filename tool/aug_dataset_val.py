import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torchvision.transforms as transforms
from tqdm import tqdm

# label transform
label_str2int = {
    'HGSC': 0,
    'EC': 1,
    'CC': 2,
    'LGSC': 3,
    'MC': 4
}

aug_img_path = {
    "origin-imgs" : "E:\\UBC-Ovarian-Cancer\\train_thumbnails",
    "augs-imgs" : "E:\\UBC-Ovarian-Cancer\\train_thumbnails_augs"
}

csv_path = "E:\\UBC-Ovarian-Cancer\\train.csv"
img_size = (224, 224)

import datetime


starttime = datetime.datetime.now()

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
                #print(f'process: {origin_imgs_file}')
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
                    augs_imgs_file = os.path.join(self.augs_imgs_path, str(img_id) + "_thumbnail" + f"_trans{j + 1}" + f"_{i + 1}.png")
                    if os.path.isfile(augs_imgs_file):
                        #print(f'process: {augs_imgs_file}')
                        img = Image.open(augs_imgs_file).convert('RGB')
                        label_int = label_str2int[f'{label}']
                        # img = cv2.imread(img_file_path)
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if self.transform:
                            img = self.transform(img)

                        self.imgs.append(img)
                        self.labels.append(label_int)

    def __len__(self):
        return len(self.imgs)

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

ubc_dataset = UBCDataset_augs(csv_path, aug_img_path, transforms)

print(ubc_dataset)
print(ubc_dataset.__len__())
endtime = datetime.datetime.now()
runtime = (endtime - starttime).seconds
print(runtime)
