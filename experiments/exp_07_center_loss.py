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
from loss import center_loss, triplet_loss
from utils import data, evaluation, kaggle_submission, logging

flags.DEFINE_float("lr", 0.0001, "Learning rate")
flags.DEFINE_string("matching_pairs", None, "Path to matching pairs pkl file")
flags.DEFINE_float("margin", 0.2, "Margin for triplet loss")
flags.DEFINE_boolean("eval", False, "Evaluation")
flags.DEFINE_float("centerloss_beta", 1, "Center loss multiplier")
flags.DEFINE_boolean("sparse", False, "Use sparse center loss")
flags.DEFINE_float("center_lr", 0.0001, "Learning rate for centerloss params")
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
    label_encoder = data.LabelEncoder()
    encoded_labels = label_encoder.fit_transform(
        df_train["landmark_id"].to_numpy()
    )
    df_train["landmark_id"] = encoded_labels
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
        sampler=SubsetRandomSampler(random.sample(list(valid_set), 100000)),
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
    if FLAGS.eval:
        df_train = pd.read_csv("./data/google-landmark/valid_train.csv")
        dataset = landmark_recognition.Dataset(
            df_train, "./data/google-landmark/train", transform
        )
        gallery_dl = DataLoader(
            dataset=dataset, batch_size=FLAGS.batch_size, num_workers=16
        )
        kaggle_submission.generate_submission(model, gallery_dl=gallery_dl)
    else:
        triplet_l = triplet_loss.OnlineHardNegativeMining(FLAGS.margin)
        center_l = center_loss.CenterLoss(
            max(dataset.labels) + 1, model.fc.out_features
        )
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": center_l.parameters(), "lr": FLAGS.center_lr},
            ],
            lr=FLAGS.lr,
        )
        center_l.to(FLAGS.device)
        losses = [(triplet_l, 1), (center_l, FLAGS.centerloss_beta)]
        loss = lambda y_pred, y: sum([a * l(y_pred, y) for l, a in losses])
        trainer = engine.create_supervised_trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss,
            device=FLAGS.device,
            non_blocking=True,
        )

        logging.attach_loggers(
            train_engine=trainer,
            eval_engine=None,
            model=model,
            early_stopping_metric="Accuracy",
        )
        trainer.run(train_dl, max_epochs=50)


if __name__ == "__main__":
    app.run(main)
