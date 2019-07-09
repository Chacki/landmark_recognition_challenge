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
from torch.nn.modules.loss import CrossEntropyLoss
import config
from dataset import landmark_recognition
from utils import evaluation, logging, data
import models
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader


flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("num_classes", 1000, "Number of Classes")
flags.DEFINE_boolean("eval", False, "Evaluation")
FLAGS = flags.FLAGS


class ClassSampler(torch.utils.data.Sampler):

    def __init__(self, data_source):
        self.first_1000_classes = np.reshape(np.argwhere(data_source < FLAGS.num_classes),(-1,))




    def __iter__(self):
        return iter(np.random.permutation(self.first_1000_classes))


    def __len__(self):
        return len(self.first_1000_classes)


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """

    label_encoder = data.LabelEncoder()
    config.init_experiment()

    df_train = pd.read_csv("./data/google-landmark/train.csv")

    transform = transforms.Compose(
        [
            transforms.Lambda(Image.open),
            transforms.Grayscale(3),
            transforms.Resize(size=(FLAGS.height, FLAGS.width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    df_train["landmark_id"] = label_encoder.fit_transform(df_train["landmark_id"].to_numpy())
    dataset = landmark_recognition.Dataset(
        df_train, "./data/google-landmark/train", transform
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        sampler=ClassSampler,
        num_workers=16
    )



    model = models.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
        device=FLAGS.device,
        non_blocking=True,
    )
    if FLAGS.eval is False:
        model.fc = nn.Linear(model.fc.in_features, FLAGS.num_classes)
        #evaluater = evaluation.build_evaluater(model, test_loader)
        #evaluation.attach_eval(evaluater, trainer, train_loader)
        #logging.attach_loggers(trainer, evaluater, model)
    trainer.run(dataloader, max_epochs=100)



if __name__ == "__main__":
    app.run(main)
