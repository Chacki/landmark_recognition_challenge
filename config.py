"""
File: config.py
Description: Define flags which should be defined for every experiment
"""
import os

from absl import flags

flags.DEFINE_string("checkpoint_dir", None, "Save checkpoints here")
flags.DEFINE_string("evaluation_dir", None, "Save evaluation artifacts here")
flags.DEFINE_string("tensorboard_dir", None, "Save tensorboard logs here")
flags.DEFINE_string("gpu", "", "Select gpu(s)")
flags.DEFINE_integer("height", 224, "Image height")
flags.DEFINE_integer("width", 224, "Image width")
FLAGS = flags.FLAGS


def init_experiment():
    """ Initialize experiment.
    - creates missing directories
    - select gpu
    """
    for directory in [
        FLAGS.checkpoint_dir,
        FLAGS.evaluation_dir,
        FLAGS.tensorboard_dir,
    ]:
        if directory:
            os.makedirs(directory, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
