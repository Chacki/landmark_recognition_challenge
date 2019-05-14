import pandas as pd
import torch
from absl import flags

FLAGS = flags.FLAGS


class Train(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.train_df = pd.read_csv(FLAGS.train_csv)
