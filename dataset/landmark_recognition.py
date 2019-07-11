from os import listdir, path

import numpy as np
import pandas as pd
import torch
from absl import flags
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

flags.DEFINE_enum(
    "dataset",
    "google-landmark",
    listdir("./data/"),
    "select dataset from ./data/",
)
FLAGS = flags.FLAGS


class Dataset(Dataset):
    """ Loads training data. csv must have two columns: 'url' and 'landmark_id'
    """

    def __init__(self, dataframe, directory, transforms):
        super().__init__()
        self.directory = directory
        self.transforms = transforms
        self.ids = dataframe["id"].to_numpy()
        self.labels = dataframe["landmark_id"].to_numpy()

    def __getitem__(self, index):
        img_name, label = self.ids[index], self.labels[index]
        img_path = path.join(self.directory, img_name + ".jpg")
        return (self.transforms(img_path), torch.from_numpy(np.array(label)))

    def __len__(self):
        return self.ids.shape[0]
