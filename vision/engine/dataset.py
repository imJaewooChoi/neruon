import os
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomDataset(Dataset):
    def __init__(self, config, phase):
        self.config = config
        self.data_dir = config["Data"]["data_dir"]
        self.phase = phase

        if phase in ["train", "valid"]:
            self.images = glob(self.data_dir + "/train/*/*.png")
            self.labels = [int(image.split("/")[-2]) for image in self.images]
            (
                self.train_images,
                self.train_labels,
                self.valid_images,
                self.valid_labels,
            ) = self._split()

            if phase == "train":
                self.images = self.train_images
                self.labels = self.train_labels
            else:
                self.images = self.valid_images
                self.labels = self.valid_labels

        elif phase == "test":
            self.images = glob(os.path.join(self.data_dir, "test/*/*.png"))
            self.labels = [int(image.split("/")[-2]) for image in self.images]

        if phase == "train":

            self.transform = A.Compose(
                [
                    A.Resize(
                        config["Data"]["image_size"], config["Data"]["image_size"]
                    ),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        elif phase == "valid":

            self.transform = A.Compose(
                [
                    A.Resize(
                        config["Data"]["image_size"], config["Data"]["image_size"]
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(
                        config["Data"]["image_size"], config["Data"]["image_size"]
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def _split(self):
        train_images, valid_images, train_labels, valid_labels = train_test_split(
            self.images,
            self.labels,
            test_size=self.config["Data"]["valid_size"],
            random_state=self.config["seed"],
        )

        return (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        return image, label


if __name__ == "__main__":
    import yaml
    import pickle

    with open("/workspace/dl_choi/vision/configs/classification.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = CustomDataset(config, "test")
    images = dataset.images
    labels = dataset.labels
    print(images[:5])
    print(labels[:5])
