import torch
from absl import app, flags
import pandas as pd
import pickle
from os import path
from PIL import Image
from torchvision import transforms
import ignite

import config
from dataset import matching_pairs_sampler, landmark_recognition
from models import resnet



flags.DEFINE_float("lr", 0.0001, "Learning rate")
flags.DEFINE_string("matching_pairs", None, "Path to matching pairs pkl file")
FLAGS = flags.FLAGS

def main(_):
    config.init_experiment()

    df_train = pd.read_csv("./data/google-landmark/valid_train.csv")
    with open(path.join("./data/google-landmark", FLAGS.matching_pairs + ".pkl"), 'rb') as f:
        dists = pickle.load(f)
    csv_path = "./data/google-landmark/valid_train.csv"
    directory = path.dirname(csv_path)
    transform = transforms.Compose([
        transforms.Lambda(Image.open),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    dataset = landmark_recognition.Dataset(df_train, directory, transform)
    sampler = matching_pairs_sampler.MatchingPairsSampler(dists)

    model = resnet.build_model(out_dim=FLAGS.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
    trainer = ignite.engine.create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        # TODO set loss to triplet loss
        #loss_fn=CrossEntropyLoss(),
        device=FLAGS.device,
        non_blocking=True,
    )
    







if __name__ == "__main__":
    app.run(main)