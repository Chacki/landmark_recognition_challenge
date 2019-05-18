from os import listdir, path

import numpy as np
import pandas as pd
import torch
from absl import flags
from PIL import Image
from torchvision import transforms

flags.DEFINE_enum(
    "dataset", None, listdir("./data/"), "select dataset from ./data/"
)
flags.mark_flag_as_required("dataset")
FLAGS = flags.FLAGS


class Train(torch.utils.data.Dataset):
    """ Loads training data. csv must have two columns: 'url' and 'landmark_id'
    """

    def __init__(self):
        super().__init__()
        csv_path = path.join("./data/", FLAGS.dataset, "train.csv")
        self.train_df = pd.read_csv(csv_path)
        self.directory = path.dirname(csv_path)
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(Image.open),
                transforms.Resize(size=(FLAGS.height, FLAGS.width)),
                transforms.ToTensor(),
            ]
        )

    @property
    def labels(self):
        """ return all labels
        """
        return self.train_df["landmark_id"].to_numpy()

    def __getitem__(self, index):
        img_path, label = (
            self.train_df[["path", "landmark_id"]].iloc[index].T.to_numpy()
        )
        img_path = path.join(path.expanduser(self.directory), img_path)
        return {
            "img": self.transforms(img_path),
            "label": torch.from_numpy(np.asarray(label)).float(),
        }

    def __len__(self):
        return self.train_df.shape[0]
