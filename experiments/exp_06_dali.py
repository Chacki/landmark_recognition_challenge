"""
File: exp_05_adaptive_softmax.py
Description: Use adaptive softmax to train on entire dataset
"""
import ignite
import numpy as np
import torch
import torch.nn as nn
from absl import app, flags
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

import config
from dataset import dali
from models import resnet
from utils import data, evaluation, logging

flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("dim", 1024, "Dimension of output vector")
FLAGS = flags.FLAGS


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()

    label_encoder = data.LabelEncoder()
    train_loader, prepare_batch = dali.get_dataloader()

    model = resnet.build_model(out_dim=FLAGS.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    loss = nn.AdaptiveLogSoftmaxWithLoss(
        in_features=FLAGS.dim, n_classes=num_labels, cutoffs=[10, 100, 1000]
    ).to(FLAGS.device)
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=lambda *x: loss(*x).loss,
        device=FLAGS.device,
        non_blocking=True,
        prepare_batch=prepare_batch,
    )
    evaluater = evaluation.build_evaluater(model, train_loader)
    evaluation.attach_eval(evaluater, trainer, train_loader)
    logging.attach_loggers(trainer, evaluater, model)
    trainer.run(train_loader, max_epochs=100)


if __name__ == "__main__":
    app.run(main)
