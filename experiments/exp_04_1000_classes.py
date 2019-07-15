"""
File: exp_04_1000_classes.py
Description: Experiment script for testing purpose only
"""
import ignite
import numpy as np
import pandas as pd
import torch
from absl import app, flags
from PIL import Image
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

import config
import models
from dataset import landmark_recognition
from utils import data, kaggle_submission, logging

flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("num_classes", 1000, "Number of Classes")
flags.DEFINE_boolean("eval", False, "Evaluation")
FLAGS = flags.FLAGS


class ClassSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.first_1000_classes = np.reshape(
            np.argwhere(data_source < FLAGS.num_classes), (-1,)
        )

    def __iter__(self):
        return iter(np.random.permutation(self.first_1000_classes))

    def __len__(self):
        return len(self.first_1000_classes)


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """

    config.init_experiment()

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: Image.open(x).convert("RGB")),
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model = models.build_model()
    models.load_checkpoint(model)
    if FLAGS.eval:
        df_gallery = pd.read_csv("./data/google-landmark/valid_train.csv")
        gallery_ds = landmark_recognition.Dataset(
            df_gallery, "./data/google-landmark/train", transform
        )
        gallery_dl = DataLoader(
            gallery_ds, batch_size=FLAGS.batch_size, num_workers=16
        )
        kaggle_submission.generate_submission(model, gallery_dl=gallery_dl)
    else:
        label_encoder = data.LabelEncoder()
        df_train = pd.read_csv("./data/google-landmark/valid_train.csv")
        encoded_labels = label_encoder.fit_transform(
            df_train["landmark_id"].to_numpy()
        )
        df_train["landmark_id"] = encoded_labels
        dataset = landmark_recognition.Dataset(
            df_train, "./data/google-landmark/train", transform
        )
        sampler = ClassSampler(encoded_labels)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=FLAGS.batch_size,
            sampler=sampler,
            num_workers=16,
        )
        train_model = nn.Sequential(
            model,
            nn.ReLU(),
            nn.Linear(model.fc.out_features, FLAGS.num_classes),
        )
        optimizer = torch.optim.Adam(train_model.parameters(), lr=FLAGS.lr)
        trainer = ignite.engine.create_supervised_trainer(
            model=train_model,
            optimizer=optimizer,
            loss_fn=CrossEntropyLoss(),
            device=FLAGS.device,
            non_blocking=True,
        )
        logging.attach_loggers(
            train_engine=trainer, eval_engine=None, model=model
        )
        trainer.run(dataloader, max_epochs=100)


if __name__ == "__main__":
    app.run(main)
