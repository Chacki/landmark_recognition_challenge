import pickle
import random
from os import path

import numpy as np
import pandas as pd
import PIL
import torch
from absl import app, flags, logging
from scipy.spatial.distance import pdist, squareform
from torchvision import transforms
from tqdm.auto import tqdm

import models

flags.DEFINE_string("checkpoint", None, "Path to checkpoint file to load")
flags.DEFINE_string("save_dir", None, "Directory to save artifacts to")
flags.DEFINE_string(
    "csv", "data/google-landmark/valid_train.csv", "Path to csv"
)
flags.DEFINE_integer("batch_size", 256, "Batch size")
FLAGS = flags.FLAGS

CSV = "data/google-landmark/valid_train.csv"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.paths = df["path"].to_numpy()
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: PIL.Image.open(x).convert("RGB")),
                transforms.RandomCrop(size=(224, 224), pad_if_needed=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, idx):
        return self.transforms(self.paths[idx])

    def __len__(self):
        return len(self.paths)


def get_feats(df):
    model = models.build_model(checkpoint=FLAGS.checkpoint)
    model.eval()
    model.to("cuda")
    dataset = Dataset(df)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=FLAGS.batch_size,
        num_workers=16,
        shuffle=False,
    )
    results = []
    with torch.no_grad():
        for img in tqdm(dataloader):
            results.append(
                torch.nn.functional.normalize(
                    model(img.cuda()).view(img.size(0), -1)
                )
                .cpu()
                .numpy()
            )
    results = np.concatenate((results), axis=0)
    model.cpu()
    return results


def get_pairs(distv, indices, threshold):
    distm = squareform(distv)
    valid_entries = np.argwhere(distm > threshold)
    if len(valid_entries) == 0:
        return []
    ids = [(indices[a], indices[b]) for a, b in np.split(valid_entries, 2)[0]]
    return random.sample(ids, 100) if len(ids) > 100 else ids


def main(_):
    logging.info(f"Reading csv: {FLAGS.csv}")
    df_csv = pd.read_csv(FLAGS.csv)
    logging.info("Extracting feats")
    feats = get_feats(df_csv)
    np.save(path.join(FLAGS.save_dir, "valid_train_feats.npy"), feats)
    df_feats = pd.DataFrame(data=feats)
    df_feats["landmark_id"] = df_csv["landmark_id"]
    logging.info("Calculating distances")
    dists = {
        x: (pdist(feats[v], np.dot), v)
        for x, v in tqdm(df_feats.groupby("landmark_id").groups.items())
    }
    print("Type threshold of cosine similarity:")
    for threshold in iter(input, ""):
        pairs = {
            x: get_pairs(v[0], v[1], float(threshold))
            for x, v in tqdm(dists.items())
        }
        pairs = {x: v for x, v in pairs.items() if len(v) > 0}
        num_pairs = [len(v) for v in pairs.values()]
        if len(num_pairs) == 0:
            print("Threshold too high 😭")
            continue
        print("Number of classes:     ", len(pairs.keys()))
        print("Number of pairs:       ", sum(num_pairs))
        print("Min samples per class: ", min(num_pairs))
        print("Mean samples per class:", sum(num_pairs) / len(pairs.keys()))
        print("Type new threshold or type enter if done")

    with open(path.join(FLAGS.save_dir, "valid_train_pairs.pkl"), "wb") as f:
        pickle.dump(pairs, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
