"""
File: sampler.py
Description: Contains sampler for triplet loss
"""
from collections import defaultdict

import numpy as np
import torch
from absl import flags
from tqdm import tqdm

FLAGS = flags.FLAGS


class RandomIdentitySampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.label2idx = defaultdict(list)
        for index, label in tqdm(
            enumerate(data_source), total=len(data_source)
        ):
            self.label2idx[label].append(index)
        self.num_instances = FLAGS.batch_size // 8
        self.labels = list(self.label2idx.keys())

    def __iter__(self):
        labels = np.random.permutation(self.labels)
        iterator = []
        for label in labels:
            idxs = self.label2idx[label]
            replace = len(idxs) < self.num_instances
            iterator.extend(
                np.random.choice(idxs, size=self.num_instances, replace=replace)
            )
        return iter(iterator)

    def __len__(self):
        return len(self.labels) * self.num_instances
