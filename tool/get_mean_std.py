from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
import os
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image


label_str2int = {
    'HGSC': 0,
    'EC': 1,
    'CC': 2,
    'LGSC': 3,
    'MC': 4
}

dataset_path = "E:\\UBC-Ovarian-Cancer\\train_thumbnails"
csv_path = "E:\\UBC-Ovarian-Cancer\\train.csv"

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



def get_mean_and_std(train_dataset):


transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


if __name__ == "__main__":
    #train_dataset = ImageFolder(root=dataset_path, transform=transforms.ToTensor())
    train_dataset = UBCDataset(csv_path=csv_path, imgs_path=dataset_path, transforms=transforms)
    train_dataset = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
    mean, std = get_mean_and_std(train_dataset)
    print(f'mean: {mean}, std: {std}')

