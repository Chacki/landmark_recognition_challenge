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


class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(
            device="cpu", image_type=types.RGB, interp_type=types.INTERP_LINEAR
        )
        self.cmn = ops.CropMirrorNormalize(
            device="cpu",
            output_dtype=types.FLOAT,
            crop=(227, 227),
            image_type=types.RGB,
            mean=[128.0, 128.0, 128.0],
            std=[1.0, 1.0, 1.0],
        )
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.resize_rng = ops.Uniform(range=(256, 480))

    def base_define_graph(self, inputs, labels):
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter=self.resize_rng())
        output = self.cmn(
            images, crop_pos_x=self.uniform(), crop_pos_y=self.uniform()
        )
        return (output, labels)


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
