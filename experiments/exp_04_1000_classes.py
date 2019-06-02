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
from torchvision.models import resnet
from torch.nn.modules.loss import CrossEntropyLoss
import config
from dataset import landmark_recognition
from utils import evaluation, logging, data
from models import resnet


flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("num_classes", 1000, "Number of Classes")
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
    train_loader, test_loader, db_loader = landmark_recognition.get_dataloaders(
        train_sampler=ClassSampler,
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
        label_encoder=label_encoder.fit_transform,
    )
    model = resnet.build_model(out_dim=FLAGS.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=CrossEntropyLoss(),
        device=FLAGS.device,
        non_blocking=True,
    )
    evaluater = evaluation.build_evaluater(model, test_loader)
    evaluation.attach_eval(evaluater, trainer, train_loader)
    logging.attach_loggers(trainer, evaluater, model)
    trainer.run(train_loader, max_epochs=100)



if __name__ == "__main__":
    app.run(main)
