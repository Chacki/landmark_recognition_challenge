from functools import partial
from os import path

import torch
import torchvision.models as models
from absl import app, flags
from torch import nn

from utils import layers

from . import resnet

__models__ = {
    "resnet50_imagenet": models.resnet50,
    "resnet50_places365": resnet.resnet50_places365,
    "resnext101_32x8d_wsl": resnet.resnext101_32x8d_wsl,
}

flags.DEFINE_enum(
    "model",
    "resnet50_imagenet",
    list(__models__.keys()),
    "Pretrained model to use",
)
flags.DEFINE_string("checkpoint", None, "Load checkpoint from given epoch")
flags.DEFINE_integer("output_dim", None, "Output dimension of feature vector")
FLAGS = flags.FLAGS


def build_model():
    # model gets an image as input and returns a vector
    model = __models__[FLAGS.model]()
    if FLAGS.output_dim is None:
        model.fc = layers.Lambda(lambda x: x.view(x.size(0), -1))
    else:
        model.fc = nn.Linear(model.fc.in_features, FLAGS.output_dim)
    if FLAGS.checkpoint:
        checkpoint_name = f"state_dict_model_{FLAGS.checkpoint}.pth"
        state_dict = torch.load(
            path.join(FLAGS.checkpoint_dir, checkpoint_name)
        )
        model.load_state_dict(state_dict)
    return model
