"""
File: download_dataset.py
Description: Hacky script to download all available images from a csv.
"""
import urllib.request
from io import BytesIO
from multiprocessing.pool import Pool
from os import makedirs, path

import numpy as np
import pandas as pd
from absl import app, flags
from PIL import Image

flags.DEFINE_string("csv", None, "path to csv with column 'url'")
flags.DEFINE_string(
    "name", None, "Name of folder in ./data/NAME to save images into."
)
flags.DEFINE_integer("num", None, "Downloads first num images")
flags.DEFINE_integer("width", 256, "width of images")
flags.DEFINE_integer("height", 256, "width of images")
flags.mark_flags_as_required(["csv", "name"])

FLAGS = flags.FLAGS


def mapping_fn(row):
    save_dir = path.join("./data/", FLAGS.name)
    try:
        save_path = path.join(
            path.splitext(path.basename(FLAGS.csv))[0], row[0] + ".jpg"
        )
        if not path.exists(path.join(save_dir, save_path)):
            result = urllib.request.urlopen(row[1])
            img = Image.open(BytesIO(result.read()))
            img.thumbnail((FLAGS.width, FLAGS.height))
            img.save(path.join(save_dir, save_path))
        else:
            print(f"Image already exists: {save_path}")
        return save_path
    except:
        print(f"URL not valid: {row[1]}")
        return None


def main(_):
    save_dir = path.join("./data/", FLAGS.name)
    makedirs(
        path.join(save_dir, path.splitext(path.basename(FLAGS.csv))[0]),
        exist_ok=True,
    )

    df = pd.read_csv(FLAGS.csv, nrows=FLAGS.num)
    pool = Pool(8)
    paths = pool.map(func=mapping_fn, iterable=df.to_numpy())
    df["path"] = np.asarray(paths)
    df = df[df["path"].notnull()]
    df.to_csv(path.join(save_dir, path.basename(FLAGS.csv)))


if __name__ == "__main__":
    app.run(main)
