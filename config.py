"""
File: config.py
Description: Define flags which should be defined for every experiment
"""
import os
import shutil

from absl import flags

flags.DEFINE_string("checkpoint_dir", None, "Save checkpoints here")
flags.DEFINE_string("evaluation_dir", None, "Save evaluation artifacts here")
flags.DEFINE_string("tensorboard_dir", None, "Save tensorboard logs here")
flags.DEFINE_string("device", "cuda", "Select device")
flags.DEFINE_integer("height", 224, "Image height")
flags.DEFINE_integer("width", 224, "Image width")
flags.DEFINE_integer("batch_size", 32, "batch size")
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
            if (
                os.path.isdir(directory)
                and len(os.listdir(directory)) > 0
                and input(f"Delete {directory} ? (y/n)") == "y"
            ):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
