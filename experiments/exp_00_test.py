"""
File: exp_00_test.py
Description: Experiment script for testing purpose only
"""
import ignite
import torch
from absl import app, flags
from ignite.contrib.handlers import ProgressBar
from torch.utils.data import DataLoader
from torchvision.models import resnet

import config
from dataset import landmark_recognition, sampler
from loss import triplet_loss

FLAGS = flags.FLAGS


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()
    train_dataset = landmark_recognition.Train()
    identity_sampler = sampler.RandomIdentitySampler(train_dataset.labels)
    train_loader = DataLoader(
        train_dataset, sampler=identity_sampler, batch_size=FLAGS.batch_size
    )
    model = resnet.resnet18(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=triplet_loss.TripletLoss(margin=1),
        prepare_batch=lambda x, device, non_blocking: (x["img"], x["label"]),
    )
    ignite.metrics.RunningAverage(output_transform=lambda x: x).attach(
        trainer, "loss"
    )
    ProgressBar(persist=True).attach(trainer, ["loss"])
    trainer.run(train_loader)
if __name__ == "__main__":
    app.run(main)
