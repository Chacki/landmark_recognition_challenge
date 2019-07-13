import math
import os

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50

from utils import layers


def resnext101_32x8d_wsl():
    model = torch.hub.load(
        "facebookresearch/WSL-Images", "resnext101_32x8d_wsl"
    )
    return model


def resnet50_places365():
    model_file = "resnet50_places365.pth.tar"
    if not os.access(model_file, os.W_OK):
        weight_url = (
            "http://places2.csail.mit.edu/models_places365/" + model_file
        )
        os.system("wget " + weight_url)

    model = resnet50(num_classes=365)
    checkpoint = torch.load(
        model_file, map_location=lambda storage, loc: storage
    )
    state_dict = {
        str.replace(k, "module.", ""): v
        for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model
