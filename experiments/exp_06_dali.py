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

flags.DEFINE_float("lr", 0.0001, "Learning rate")
flags.DEFINE_integer("dim", 2048, "Dimension of output vector")
FLAGS = flags.FLAGS

from itertools import chain

def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()

    label_encoder = data.LabelEncoder()
    train_loader, prepare_batch, labels = dali.get_dataloader()

    backbone = resnet.build_model(out_dim=FLAGS.dim)
    output_layer = nn.Linear(FLAGS.dim, max(labels)+1)
    model = nn.Sequential(backbone, nn.ReLU(), output_layer)
    for part in backbone[:-1]:
        for param in part.parameters():
            param.requires_grad = False
    optimizer = torch.optim.Adam(chain(backbone[-1].parameters(), output_layer.parameters()), lr=FLAGS.lr)
    #loss = nn.AdaptiveLogSoftmaxWithLoss(
    #    in_features=FLAGS.dim, n_classes=max(labels) + 1, cutoffs=[10, 100, 1000]
    #).to(FLAGS.device)
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        #loss_fn=lambda *x: loss(*x).loss,
        loss_fn=nn.CrossEntropyLoss(),
        device=FLAGS.device,
        non_blocking=True,
        prepare_batch=prepare_batch,
    )
    logging.attach_loggers(trainer, None, model)
    trainer.run(train_loader, max_epochs=100)


if __name__ == "__main__":
    app.run(main)
