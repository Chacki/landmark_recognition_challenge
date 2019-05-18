import urllib.request
from io import BytesIO
from multiprocessing.pool import Pool
from os import path

import numpy as np
import pandas as pd
from absl import app, flags
from PIL import Image

flags.DEFINE_string("csv", "", "path to csv with column 'url'")
flags.DEFINE_string("dest", "", "destination_dir")
flags.DEFINE_integer("width", 256, "width of images")
flags.DEFINE_integer("height", 256, "width of images")

FLAGS = flags.FLAGS


def mapping_fn(row):
    try:
        result = urllib.request.urlopen(row[1])
        img = Image.open(BytesIO(result.read()))
        img.thumbnail((FLAGS.width, FLAGS.height))
        img.save(path.join(FLAGS.dest, row[0] + ".jpg"))
        return row[0] + ".jpg"

    except:
        return None


def main(_):
    df = pd.read_csv(FLAGS.csv, nrows=1000)
    pool = Pool(8)
    paths = pool.map(func=mapping_fn, iterable=df.to_numpy())
    df["path"] = np.asarray(paths)

    df = df[df["path"].notnull()]
    df.to_csv(path.join(FLAGS.dest, path.basename(FLAGS.csv)))


if __name__ == "__main__":
    app.run(main)
