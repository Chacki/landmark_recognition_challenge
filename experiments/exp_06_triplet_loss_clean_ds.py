import pickle
import random
from os import path

import pandas as pd
import torch
from absl import app, flags
from ignite import engine, handlers, metrics
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

import config
import models
from dataset import landmark_recognition, matching_pairs_sampler
from loss import triplet_loss
from utils import evaluation, logging

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
    pairs = {k: v for k, v in pairs.items() if len(v) > 0}
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: Image.open(x).convert("RGB")),
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = landmark_recognition.Dataset(
        df_train, "./data/google-landmark/train", transform
    )
    train_set = set([idx for ps in pairs.values() for p in ps for idx in p])
    all_set = set(range(len(dataset.labels)))
    valid_set = all_set - train_set
    print("Train set length: ", len(train_set))
    print("Valid set length: ", len(valid_set))
    valid_dl = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        sampler=SubsetRandomSampler(random.sample(list(valid_set), 10000)),
        num_workers=16,
    )
    train_sampler = matching_pairs_sampler.MatchingPairsSampler(pairs)
    train_dl = DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        num_workers=16,
    )

    model = models.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    trainer = engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=triplet_loss.OnlineHardNegativeMining(FLAGS.margin),
        device="cuda",
        non_blocking=True,
    )
    acc_metric = evaluation.CalculateAccuracy(train_dl, model)
    evaluater = engine.create_supervised_evaluator(
        model=model,
        metrics={"Accuracy": metrics.EpochMetric(acc_metric)},
        device="cuda",
    )
    # trainer.add_event_handler(
    #     engine.Events.EPOCH_COMPLETED, lambda x: evaluater.run(valid_dl)
    # )

    logging.attach_loggers(
        train_engine=trainer,
        eval_engine=None,
        model=model,
        early_stopping_metric="Accuracy",
        additional_tb_log_handler=[
            (
                logging.EmbeddingHandler(model, valid_dl),
                engine.Events.EPOCH_COMPLETED,
            )
        ],
    )
    trainer.run(train_dl, max_epochs=50)


if __name__ == "__main__":
    app.run(main)
