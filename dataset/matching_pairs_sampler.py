import random

from torch.utils.data import Sampler


class MatchingPairsSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = []
        # shuffle values randomly
        random_values = list(self.data_source.values())
        random.shuffle(random_values)
        # take randomly one pair for each id and append it to indices list
        for value in random_values:
            random_pair = random.sample(value, 1)[0]
            indices.extend(random_pair)
        return iter(indices)

    def __len__(self):
        return 2 * len(self.data_source)
