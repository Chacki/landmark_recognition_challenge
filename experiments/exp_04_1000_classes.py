"""
File: exp_00_test.py
Description: Experiment script for testing purpose only
"""
import ignite
import torch
import numpy as np
from absl import app, flags
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet
from torch.nn.modules.loss import CrossEntropyLoss
import config
from dataset import landmark_recognition, sampler
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from utils import evaluation, logging
from collections import defaultdict

flags.DEFINE_float("lr", 0.001, "Learning rate")
FLAGS = flags.FLAGS


class ClassSampler(torch.utils.data.Sampler):

    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source

        self._num_samples = num_samples


        if self._num_samples is not None:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

        self.label2idx = defaultdict(list)
        for index, label in tqdm(
                enumerate(data_source), total=len(data_source)
        ):
            self.label2idx[label].append(index)
        self.labels = list(self.label2idx.keys())
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(self.labels)
        unique_labels, label_count = np.unique(self.labels, return_counts=True)
        # labels with respect to number of occurence
        self.sorted_label2label = unique_labels[np.flip(np.argsort(label_count))]
        self.label2slabel = np.argsort(self.sorted_label2label)
        transformed_labels = self.label2slabel[self.labels]
        first_1000_classes = np.argwhere(transformed_labels < 1000)
        self.num_instances = len(data_source)
        return first_1000_classes

    

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n).tolist())


    def __len__(self):
        return self.num_samples


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()
    train_loader, test_loader, db_loader = landmark_recognition.get_dataloaders(
        train_sampler=ClassSampler,
        transforms=transforms.Compose(
            [
                transforms.Lambda(Image.open),
                transforms.Grayscale(3),
                transforms.Resize(size=(FLAGS.height, FLAGS.width)),
                transforms.ToTensor(),
            ]
        ),
    )
    model = resnet.resnet50(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
        device=FLAGS.device,
        non_blocking=True,
    )
    evaluater = evaluation.build_evaluater(model, test_loader)
    evaluation.attach_eval(evaluater, trainer, train_loader)
    logging.attach_loggers(trainer, evaluater, model)
    trainer.run(train_loader, max_epochs=100)


if __name__ == "__main__":
    app.run(main)
