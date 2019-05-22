"""
File: exp_00_test.py
Description: Experiment script for testing purpose only
"""
import ignite
import torch
from absl import app, flags
from torchvision.models import resnet

import config
from dataset import landmark_recognition, sampler
from loss import triplet_loss
from utils import evaluation, logging

flags.DEFINE_float("margin", 0.3, "margin fro triplet loss")
FLAGS = flags.FLAGS


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()
    train_loader, test_loader, db_loader = landmark_recognition.get_dataloaders(
        sampler.RandomIdentitySampler
    )
    model = resnet.resnet50(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=triplet_loss.TripletLoss(margin=FLAGS.margin),
        device=FLAGS.device,
        non_blocking=True,
    )
    evaluater = evaluation.build_validator(model, test_loader)
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED, lambda _: evaluater.run(db_loader)
    )
    logging.attach_loggers(trainer, evaluater, model)
    trainer.run(train_loader, max_epochs=100)
if __name__ == "__main__":
    app.run(main)
