from os import path

import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
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
