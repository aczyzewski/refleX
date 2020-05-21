from __future__ import print_function, division

import os
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset


class RefleXDataset(Dataset):
    """ RefleX datasets """

    def __init__(self, csv_file: str, root_dir: str, size: int = 1024,
                 use_augmentation: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            augumentation (bool): Turns on additional augumentation
                of an image
        """

        self.labels = pd.read_csv(csv_file)
        self.size = size
        self.root_dir = root_dir
        self.tfms = [
            transforms.ToTensor(),

            # This numbers have been calclulated using `calculate_mean_std`
            # method from `data` module on images in `labels_train.csv` file.
            # (https://zenodo.org/record/2605120)
            transforms.Normalize(mean=[0.7811], std=[0.7926]),
        ]

        if use_augmentation:
            raise NotImplementedError()
            self.tfms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ] + self.tfms

        # Compose all the transformations
        self.transform = transforms.Compose(self.tfms)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrive the image
        img_name = self.labels[idx, 0] + '.png'
        img_path = os.path.join(self.root_dir, str(self.size), img_name)
        raw_image = np.array(Image.open(img_path))
        image = self.transform(raw_image)

        # Retrieve the labels
        labels = self.lables.iloc[idx, 1:]
        labels = np.array([labels]).astype('float').reshape(-1, 7)
        labels = torch.from_numpy(labels)

        return image, labels
