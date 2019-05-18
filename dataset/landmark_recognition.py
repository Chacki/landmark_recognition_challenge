import pandas as pd
from os import path
import torch
from absl import flags
from PIL import Image
from torchvision import transforms

FLAGS = flags.FLAGS
flags.DEFINE_string("train_csv", "", "")


class Train(torch.utils.data.Dataset):
    """ Loads training data. csv must have two columns: 'url' and 'landmark_id'
    """

    def __init__(self):
        super().__init__()
        self.train_df = pd.read_csv(FLAGS.train_csv)
        self.directory = path.dirname(FLAGS.train_csv)
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
            "label": torch.tensor(label, dtype=torch.float),
        }

    def __len__(self):
        return self.train_df.shape[0]
