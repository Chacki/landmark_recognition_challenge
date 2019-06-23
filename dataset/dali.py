from os import listdir, path

from absl import flags

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import pandas as pd

flags.DEFINE_enum(
    "dataset", None, listdir("./data/"), "select dataset from ./data/"
)
flags.mark_flag_as_required("dataset")
FLAGS = flags.FLAGS


class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode = ops.HostDecoder(device="cpu", output_type=types.RGB)
        self.res = ops.Resize(device="cpu", resize_shorter=256) 
        self.cmn = ops.CropMirrorNormalize(
            device="cpu",
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            crop=(224, 224),
            image_type=types.RGB,
            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
            std=[0.229 * 255,0.224 * 255,0.225 * 255],
        )
        self.uniform = ops.Uniform(range=(0.0, 1.0))

    def base_define_graph(self, inputs, labels):
        images = self.decode(inputs)
        images = self.res(images)
        output = self.cmn(
            images, crop_pos_x=self.uniform(), crop_pos_y=self.uniform()
        )
        return (output, labels)


class MXNetReaderPipeline(CommonPipeline):
    def __init__(self):
        super(MXNetReaderPipeline, self).__init__(FLAGS.batch_size, 4, 0)
        ds_dir = path.join("./data/", FLAGS.dataset)
        self.labels = pd.read_csv(path.join(ds_dir, "blub.lst"), sep="\t", header=None)[1]
        self.input = ops.MXNetReader(
            path=[path.join(ds_dir, "blub.rec")],
            index_path=[path.join(ds_dir, "blub.idx")],
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
    labels = pipe.labels
    epochs = 10000
    class iter_with_len(DALIClassificationIterator):
        def __len__(self):
            return epochs
    dali_iter = iter_with_len([pipe], size=epochs*FLAGS.batch_size, auto_reset=True)

    def prepare_batch(batch, device, non_blocking):
        img = batch[0]["data"].to(device, non_blocking=non_blocking)
        label = (
            batch[0]["label"]
            .squeeze()
            .to(device, non_blocking=non_blocking)
            .long()
        )
        return img, label

    return dali_iter, prepare_batch, labels
