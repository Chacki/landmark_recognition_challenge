"""
Description: Entry point of this repoitory. Helps starting experiments and
             defines appropriate folder names for checkpoints and logs based
             on custom flags
"""
import os
import sys
from glob import glob
from os import path

LOG_DIR = "./log/"
CHECKPOINT_DIR = "./checkpoint/"
EVALUATION_DIR = "./evaluation/"
TENSORBOARD_DIR = "./tensorboard/"
# Following flags are ignored for log and checkpoint folder name.
FILTERED_FLAGS = ["train", "eval", "device", "epochs"]


def main(experiment_path):
    """ main entrypoint

    :experiment_path: path of python script to run
    """
    print(f"Run experiment script: {experiment_path}")
    sorted_flags = sorted(
        filter(
            lambda x: x.split("=")[0][2:] not in FILTERED_FLAGS, sys.argv[2:]
        )
    )
    name = experiment_path.split("/")[-1].split(".")[0][4:]
    changed_flags = "".join(sorted_flags)
    exp_log_dir = path.join(LOG_DIR, name + changed_flags)
    exp_checkpoint_dir = path.join(CHECKPOINT_DIR, name + changed_flags)
    exp_evaluation_dir = path.join(EVALUATION_DIR, name + changed_flags)
    exp_tensorboard_dir = path.join(TENSORBOARD_DIR, name + changed_flags)
    args = []
    args.append("--log_dir=" + exp_log_dir)
    args.append("--checkpoint_dir=" + exp_checkpoint_dir)
    args.append("--evaluation_dir=" + exp_evaluation_dir)
    args.append("--tensorboard_dir=" + exp_tensorboard_dir)
    args.append(" ".join(sys.argv[2:]))
    command = f"python -m experiments.exp_{name} " + " ".join(args).strip()
    print(f"Execute: {command}")
    return os.system(command)


if __name__ == "__main__":
    main(glob(path.join("experiments", "exp_" + sys.argv[1] + "*.py"))[0])
