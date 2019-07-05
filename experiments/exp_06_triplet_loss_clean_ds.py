import pickle
from os import path

import pandas as pd
import torch
from absl import app, flags
from ignite import contrib, engine, handlers, metrics
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import config
import models
from dataset import landmark_recognition, matching_pairs_sampler
from loss import triplet_loss

flags.DEFINE_float("lr", 0.0001, "Learning rate")
flags.DEFINE_string("matching_pairs", None, "Path to matching pairs pkl file")
FLAGS = flags.FLAGS


def main(_):
    config.init_experiment()

    df_train = pd.read_csv("./data/google-landmark/valid_train.csv")
    with open(
        path.join("./data/google-landmark", FLAGS.matching_pairs + ".pkl"), "rb"
    ) as f:
        dists = pickle.load(f)
    csv_path = "./data/google-landmark/valid_train.csv"
    directory = path.dirname(csv_path)
    transform = transforms.Compose(
        [
            transforms.Lambda(Image.open),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    dataset = landmark_recognition.Dataset(df_train, directory, transform)
    sampler = matching_pairs_sampler.MatchingPairsSampler(dists)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        sampler=sampler,
        num_workers=16,
    )

    model = models.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    trainer = engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=triplet_loss.OnlineHardNegativeMining(0.2),
        device=FLAGS.device,
        non_blocking=True,
    )
    model_checkpoint = handlers.ModelCheckpoint(
        dirname=FLAGS.checkpoint_dir,
        filename_prefix="state_dict",
        save_interval=1,
        n_saved=10,
    )
    trainer.add_event_handler(
        engine.Events.EPOCH_COMPLETED, model_checkpoint, {"model": model}
    )
    metrics.RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    contrib.handlers.ProgressBar().attach(trainer, ["loss"])
    trainer.run(dataloader, max_epochs=10)
if __name__ == "__main__":
    app.run(main)
