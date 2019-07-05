import torch
from absl import app, flags
from torch import nn

from utils import layers

from . import resnet

__models__ = {
    "resnet50_imagenet": resnet.resnet50,
    "resnet50_places365": resnet.resnet50_places365,
    "resnext101_32x8d_wsl": resnet.resnext101_32x8d_wsl,
}

flags.DEFINE_enum(
    "model",
    "resnet50_imagenet",
    list(__models__.keys()),
    "Pretrained model to use",
)
flags.DEFINE_integer("output_dim", None, "Output dimension of feature vector")
FLAGS = flags.FLAGS


def build_model(checkpoint=None, **kwargs):
    # model gets an image as input and returns a vector
    model = __models__[FLAGS.model](kwargs)
    if FLAGS.output_dim is None:
        model.fc = layers.Lambda(lambda x: x.view(x.size(0), -1))
    else:
        model.fc = nn.Linear(model.fc.in_features, FLAGS.output_dim)
    if checkpoint:
        state_dict = torch.load(checkpoint)
        out_features = state_dict["fc.weight"].size(0)
        model.fc = nn.Linear(model.fc.in_features, out_features)
        model.load_state_dict(torch.load(checkpoint))
    return model
