from os import path

import pandas as pd
from absl import app, flags

from utils import data

flags.DEFINE_string("csv", None, "Path to csv")
flags.DEFINE_bool("train", False, "csv is for training")
flags.DEFINE_bool("test", False, "csv is for testing")
flags.DEFINE_integer(
    "min_instances", 50, "Number of minimal instances each class should have"
)
FLAGS = flags.FLAGS


def prepare_train():
    label_encoder = data.LabelEncoder()
    (
        pd.read_csv(FLAGS.csv)
        .assign(
            path=lambda x: path.basename(FLAGS.csv).split(".")[0]
            + "/"
            + x.id
            + ".jpg"
        )
        .where(
            lambda x: x.path.apply(
                lambda x: path.isfile(path.join(path.dirname(FLAGS.csv), x))
            )
        )
        .groupby("landmark_id")
        .filter(lambda x: len(x) > FLAGS.min_instances)
        .assign(label=lambda x: label_encoder.fit_transform(x.landmark_id))
        .reset_index(drop=True)
        .to_csv(
            path.splitext(FLAGS.csv)[0] + ".lst",
            header=False,
            sep="\t",
            columns=["label", "path"],
        )
    )


def prepare_test():
    (
        pd.read_csv(FLAGS.csv)
        .assign(
            path=lambda x: path.basename(FLAGS.csv).split(".")[0]
            + "/"
            + x.id
            + ".jpg"
        )
        .to_csv(
            path.splitext(FLAGS.csv)[0] + ".lst",
            header=False,
            sep="\t",
            columns=["id", "path"],
        )
    )


def main(_):
    if FLAGS.train:
        prepare_train()
    elif FLAGS.test:
        prepare_test()


if __name__ == "__main__":
    app.run(main)
