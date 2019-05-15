import urllib
from io import BytesIO

import pandas as pd
import torch
from absl import flags
from PIL import Image
from torchvision import transforms

FLAGS = flags.FLAGS


class Train(torch.utils.data.Dataset):
    """ Loads training data. csv must have two columns: 'url' and 'landmark_id'
    """

    def __init__(self):
        super().__init__()
        self.train_df = pd.read_csv(FLAGS.train_csv)
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: Image.open(
                        BytesIO(urllib.request.urlopen(x).read())
                    )
                ),
                transforms.Resize(size=(FLAGS.height, FLAGS.width)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        url, label = (
            self.train_df[["url", "landmark_id"]].iloc(index).T.to_numpy()
        )
        return {"img": self.transforms(url), "label": label}

    def __len__(self):
        return self.train_df.shape[0]
