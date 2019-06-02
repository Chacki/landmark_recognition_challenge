from os import listdir, path

from absl import flags

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

flags.DEFINE_enum(
    "dataset", None, listdir("./data/"), "select dataset from ./data/"
)
flags.mark_flag_as_required("dataset")
FLAGS = flags.FLAGS


class MXNetReaderPipeline(CommonPipeline):
    def __init__(self):
        super(MXNetReaderPipeline, self).__init__(FLAGS.batch_size, 4, 0)
        ds_dir = path.join("./data/")
        self.input = ops.MXNetReader(
            path=[path.join(ds_dir, "train.rec")],
            index_path=[path.join(ds_dir, "train.idx")],
            random_shuffle=True,
            shard_id=0,
            num_shards=1,
        )

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)


def get_dataloader():
    pipe = MXNetReaderPipeline()
    pipe.build()
    dali_iter = DALIClassificationIterator([pipe], pipe.epoch_size("Reader"))

    def prepare_batch(batch, device, non_blocking):
        img = batch[0]["data"].to(device, non_blocking=non_blocking)
        label = (
            batch[0]["label"]
            .squeeze()
            .to(device, non_blocking=non_blocking)
            .long()
        )
        return img, label

    return dali_iter, prepare_batch
