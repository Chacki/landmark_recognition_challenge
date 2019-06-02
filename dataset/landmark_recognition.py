from os import listdir, path

import numpy as np
import pandas as pd
import torch
from absl import flags
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

flags.DEFINE_enum(
    "dataset", None, listdir("./data/"), "select dataset from ./data/"
)
flags.mark_flag_as_required("dataset")
FLAGS = flags.FLAGS


class Dataset(Dataset):
    """ Loads training data. csv must have two columns: 'url' and 'landmark_id'
    """

    def __init__(self, dataframe, directory, transforms):
        super().__init__()
        self.dataframe = dataframe
        self.directory = directory
        self.transforms = transforms

    @property
    def labels(self):
        """ return all labels
        """
        return self.dataframe["landmark_id"].to_numpy()

    def __getitem__(self, index):
        img_path, label = (
            self.dataframe[["path", "landmark_id"]].iloc[index].T.to_numpy()
        )
        img_path = path.join(path.expanduser(self.directory), img_path)
        return (self.transforms(img_path), torch.from_numpy(np.array(label)))

    def __len__(self):
        return self.dataframe.shape[0]


def get_dataloaders(train_sampler, transforms, label_encoder=lambda x: x):
    csv_path = path.join("./data/", FLAGS.dataset, "train.csv")
    directory = path.dirname(csv_path)
    # TODO should be filtered before downloading the images
    dataframe = (
        pd.read_csv(csv_path)
        .groupby("landmark_id")
        .filter(lambda x: len(x) > 3)
    )
    dataframe["landmark_id"] = label_encoder(dataframe["landmark_id"])
    # take 2 instances of each class for testing
    train_df, test_df = train_test_split(
        dataframe,
        test_size=int(dataframe["landmark_id"].nunique()) * 2,
        stratify=dataframe["landmark_id"],
    )
    train_set = Dataset(train_df, directory, transforms)
    test_set = Dataset(test_df, directory, transforms)
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler(train_set.labels),
        batch_size=FLAGS.batch_size,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=FLAGS.batch_size,
        pin_memory=True,
        num_workers=4,
    )
    db_loader = DataLoader(
        train_set,
        shuffle=False,
        batch_size=FLAGS.batch_size,
        pin_memory=True,
        num_workers=4,
    )
    return train_loader, test_loader, db_loader
