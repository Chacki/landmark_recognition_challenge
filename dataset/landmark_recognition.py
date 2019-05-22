from os import listdir, path

import numpy as np
import pandas as pd
import torch
from absl import flags
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

flags.DEFINE_enum(
    "dataset", None, listdir("./data/"), "select dataset from ./data/"
)
flags.mark_flag_as_required("dataset")
FLAGS = flags.FLAGS


class Dataset(torch.utils.data.Dataset):
    """ Loads training data. csv must have two columns: 'url' and 'landmark_id'
    """

    def __init__(self, dataframe, directory):
        super().__init__()
        self.dataframe = dataframe
        self.directory = directory
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(Image.open),
                transforms.Grayscale(3),
                transforms.Resize(size=(FLAGS.height, FLAGS.width)),
                transforms.ToTensor(),
            ]
        )

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
        return (
            self.transforms(img_path),
            torch.from_numpy(np.array(label)).float(),
        )

    def __len__(self):
        return self.dataframe.shape[0]


def get_dataloaders(sampler):
    csv_path = path.join("./data/", FLAGS.dataset, "train.csv")
    directory = path.dirname(csv_path)
    # TODO should be filtered before downloading the images
    dataframe = (
        pd.read_csv(csv_path)
        .groupby("landmark_id")
        .filter(lambda x: len(x) > 5)
    )
    train_df, test_df = train_test_split(
        dataframe, test_size=0.1, stratify=dataframe["landmark_id"]
    )
    train_set = Dataset(train_df, directory)
    test_set = Dataset(test_df, directory)
    train_loader = DataLoader(
        train_set,
        sampler=sampler(train_set.labels),
        batch_size=FLAGS.batch_size,
        pin_memory=True,
        num_workers=4,
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
