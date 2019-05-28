"""
File: exp_01_tripletloss.py
Description: Experiment script for triplet loss with pretrained models on
             imagenet or places365
"""
import ignite
import torch
from absl import app, flags
from PIL import Image
from torchvision import transforms

import config
from dataset import landmark_recognition, sampler
from loss import triplet_loss
from models import resnet
from utils import evaluation, logging

flags.DEFINE_float("margin", 0.3, "margin fro triplet loss")
flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("dim", 1024, "Dimension of output vector")
FLAGS = flags.FLAGS


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()
    train_loader, test_loader, db_loader = landmark_recognition.get_dataloaders(
        train_sampler=sampler.RandomIdentitySampler,
        transforms=transforms.Compose(
            [
                transforms.Lambda(Image.open),
                transforms.Grayscale(3),
                transforms.Resize(size=(FLAGS.height, FLAGS.width)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    model = resnet.build_model(out_dim=FLAGS.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
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
