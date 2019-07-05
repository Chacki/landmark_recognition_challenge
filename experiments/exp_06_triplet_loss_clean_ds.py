import pickle
from os import path

import pandas as pd
import torch
from absl import app, flags
from ignite import contrib, engine, handlers, metrics
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

import config
import models
from dataset import landmark_recognition, matching_pairs_sampler
from loss import triplet_loss

flags.DEFINE_float("lr", 0.0001, "Learning rate")
flags.DEFINE_string("matching_pairs", None, "Path to matching pairs pkl file")
flags.DEFINE_float("margin", 0.2, "Margin for triplet loss")
FLAGS = flags.FLAGS


def main(_):
    config.init_experiment()

    df_train = pd.read_csv("./data/google-landmark/valid_train.csv")
    with open(
        path.join("./data/google-landmark", FLAGS.matching_pairs + ".pkl"), "rb"
    ) as f:
        pairs = pickle.load(f)
    csv_path = "./data/google-landmark/valid_train.csv"
    directory = path.dirname(csv_path)
    transform = transforms.Compose(
        [
            transforms.Lambda(Image.open),
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = landmark_recognition.Dataset(df_train, directory, transform)
    print(len(set([v for t in pairs.values() for v in t])))
    set_valid = set()
    while len(set_valid) < 100000:
        for ps in pairs.values():
            idx1, idx2 = ps.pop()
            set_valid.add(idx1)
            set_valid.add(idx2)
        pairs = {k:v for k, v in pairs.items() if len(v) > 0}
    valid_dl = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        sampler=SubsetRandomSampler(list(set_valid)),
        num_workers=16,
    )
    train_sampler = matching_pairs_sampler.MatchingPairsSampler(pairs)
    train_dl = DataLoader(
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
        loss_fn=triplet_loss.OnlineHardNegativeMining(FLAGS.margin),
        device=FLAGS.device,
        non_blocking=True,
    )
    evaluater = engine.create_supervised_evaluator(
        model=model,
        metrics={"triplet_loss": triplet_loss.OnlineHardMining(FLAGS.margin)}
        device=FLAGS.device,
    )
    pbar = contrib.handlers.ProgressBar()
    pbar.attach(trainer, ["loss"])

    def evaluation(engine):
        evaluater.run(valid_dl)
        pbar.log_message(evaluater.state.metrics)

    trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, evaluation)

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
    trainer.run(train_dl, max_epochs=10)


if __name__ == "__main__":
    app.run(main)
