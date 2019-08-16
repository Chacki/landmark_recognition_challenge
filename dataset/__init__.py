from absl import flags
from os import listdir

flags.DEFINE_enum(
    "dataset", "google-landmark", listdir("./data/"), "select dataset from ./data/"
)
