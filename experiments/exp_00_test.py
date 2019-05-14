"""
File: exp_00_test.py
Description: Experiment script for testing purpose only
"""
import ignite
from absl import app, flags

import config

FLAGS = flags.FLAGS


def main(_):
    """ main entrypoint, must be called with app.run(main) to define all flags
    """
    config.init_experiment()


if __name__ == "__main__":
    app.run(main)
