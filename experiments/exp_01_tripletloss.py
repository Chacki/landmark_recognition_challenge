"""
File: exp_01_tripletloss.py
Description: Experiment script for triplet loss with pretrained models on
             imagenet or places365
"""
import cv2
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

flags.DEFINE_float("margin", 1, "margin fro triplet loss")
flags.DEFINE_float("lr", 0.0001, "Learning rate")
flags.DEFINE_integer("dim", 1024, "Dimension of output vector")
FLAGS = flags.FLAGS


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet.build_model(out_dim=FLAGS.dim)
        self.pooling = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, inp):
        batch_size, crops_num = inp.size(0), inp.size(1)
        inp = inp.flatten(start_dim=0, end_dim=1)
        feats = self.model(inp).view(batch_size, crops_num, -1)
        feats = feats.transpose(1, 2)
        feats = self.pooling(feats)
        feats = feats.flatten(start_dim=1)
        return feats


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()
    transforms_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_loader, test_loader, db_loader = landmark_recognition.get_dataloaders(
        train_sampler=sampler.RandomIdentitySampler,
        transforms=transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: cv2.cvtColor(
                        cv2.imread(x, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                    )
                ),
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.FiveCrop(224),
                transforms.Lambda(
                    lambda crops: torch.stack(
                        [transforms_tensor(crop) for crop in crops]
                    )
                ),
            ]
        ),
    )
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=triplet_loss.TripletLoss(margin=FLAGS.margin),
        device=FLAGS.device,
        non_blocking=True,
    )
    evaluater = evaluation.build_evaluater(model, test_loader)
    evaluation.attach_eval(evaluater, trainer, train_loader)
    logging.attach_loggers(trainer, evaluater, model)
    trainer.run(train_loader, max_epochs=100)


if __name__ == "__main__":
    app.run(main)
