"""
File: exp_05_adaptive_softmax.py
Description: Use adaptive softmax to train on entire dataset
"""
import numpy as np
import ignite
import torch
import torch.nn as nn
from absl import app, flags
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet
from models import resnet

import config
from dataset import landmark_recognition, sampler
from utils import evaluation, logging
from sklearn.preprocessing import LabelEncoder

flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("dim", 1024, "Dimension of output vector")
FLAGS = flags.FLAGS


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()
    label_encoder = LabelEncoder()

    class EncodeLabels:
        def __init__(self):
            self.label_encoder = LabelEncoder()
            self.slabel2label = None
            self.label2slabel = None

        def fit_transform(self, labels):
            labels = label_encoder.fit_transform(labels)
            unique_labels, label_count = np.unique(labels, return_counts=True)
            # labels with respect to number of occurence
            self.slabel2label = unique_labels[np.flip(np.argsort(label_count))]
            self.label2slabel = np.argsort(self.slabel2label)
            transformed_labels = self.label2slabel[labels]
            return transformed_labels

        def inverse_transform(self, labels):
            labels = self.slabel2label[labels]
            orig_labels = self.label_encoder.inverse_transform(labels)
            return orig_labels

    encode_labels = EncodeLabels()

    train_loader, test_loader, db_loader = landmark_recognition.get_dataloaders(
        train_sampler=torch.utils.data.RandomSampler,
        transforms=transforms.Compose(
            [
                transforms.Lambda(Image.open),
                transforms.Grayscale(3),
                transforms.Resize(size=max(FLAGS.height, FLAGS.width)),
                transforms.RandomCrop((FLAGS.height, FLAGS.width)),
                transforms.ToTensor(),
            ]
        ),
        label_encoder=encode_labels.fit_transform,
    )
    num_labels = len(np.unique(train_loader.dataset.labels))

    model = resnet.build_model(out_dim=FLAGS.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    loss = nn.AdaptiveLogSoftmaxWithLoss(
        in_features=FLAGS.dim, n_classes=num_labels, cutoffs=[10]
    )
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lambda pred, y: loss(pred, y.to(torch.long)).loss,
        device=FLAGS.device,
        non_blocking=True,
    )
    evaluater = evaluation.build_evaluater(model, test_loader)
    evaluation.attach_eval(evaluater, trainer, train_loader)
    logging.attach_loggers(trainer, evaluater, model)
    trainer.run(train_loader, max_epochs=100)


if __name__ == "__main__":
    app.run(main)
