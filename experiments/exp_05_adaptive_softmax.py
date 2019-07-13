"""
File: exp_05_adaptive_softmax.py
Description: Experiment script for testing purpose only
"""
import math
from itertools import chain

import ignite
import numpy as np
import pandas as pd
import torch
from absl import app, flags
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

import config
import models
from dataset import landmark_recognition
from utils import data, kaggle_submission, logging

flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_boolean("eval", False, "Evaluation")
FLAGS = flags.FLAGS


class Model(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.prediction = nn.AdaptiveLogSoftmaxWithLoss(
            self.backbone.fc.out_features,
            num_classes,
            cutoffs=[
                2 ** i for i in range(11, math.floor(math.log2(num_classes)))
            ],
            div_value=2,
        )

    def forward(self, x, target=None):
        feats = self.backbone(x)
        if self.training:
            return feats
        else:
            return self.prediction.log_prob(feats)

    def loss(self, y_pred, y):
        return self.prediction(y_pred, y).loss


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """

    label_encoder = data.LabelEncoder()
    config.init_experiment()

    df_train = pd.read_csv("./data/google-landmark/valid_train.csv")

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: Image.open(x).convert("RGB")),
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    encoded_labels = label_encoder.fit_transform(
        df_train["landmark_id"].to_numpy()
    )
    df_train["landmark_id"] = encoded_labels
    dataset = landmark_recognition.Dataset(
        df_train, "./data/google-landmark/train", transform
    )

    backbone = models.build_model()
    model = Model(backbone, max(dataset.labels) + 1)
    models.load_checkpoint(model)
    if FLAGS.eval:
        kaggle_submission.generate_submission(
            model=model, label_transforms=label_encoder.inverse_transform
        )
    else:
        train_idxs, test_idxs = train_test_split(
            np.arange(len(dataset)),
            test_size=0.05,
            shuffle=True,
            stratify=dataset.labels,
        )
        train_dl = DataLoader(
            dataset=dataset,
            batch_size=FLAGS.batch_size,
            sampler=SubsetRandomSampler(train_idxs),
            num_workers=16,
        )
        test_dl = DataLoader(
            dataset=dataset,
            batch_size=FLAGS.batch_size,
            sampler=SubsetRandomSampler(test_idxs),
            num_workers=16,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
        print(model)
        trainer = ignite.engine.create_supervised_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=model.loss,
            device=FLAGS.device,
            non_blocking=True,
        )
        evaluater = ignite.engine.create_supervised_evaluator(
            model=model,
            metrics={"Accuracy": ignite.metrics.Accuracy()},
            device=FLAGS.device,
            non_blocking=True,
        )
        trainer.add_event_handler(
            ignite.engine.Events.EPOCH_COMPLETED,
            lambda _: evaluater.run(test_dl),
        )
        logging.attach_loggers(
            train_engine=trainer,
            eval_engine=evaluater,
            model=model,
            early_stopping_metric="Accuracy",
        )
        trainer.run(train_dl, max_epochs=100)


if __name__ == "__main__":
    app.run(main)
