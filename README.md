# [Landmark Recognition Challenge](https://www.kaggle.com/c/landmark-recognition-2019)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Directories
- `./data/` contains the datasets.
- `./evaluation/` contains the evaluation artifacts, such as evaluated metrics.
- `./tensorboard/` contains the tensorboard logs.
- `./log/` contains logs, such as logged stdout.
- `./experiments/` contains experiment scripts.
    - experiment scripts must be named after the following structure: `exp_{ID}_{NAME}.py`

## Datasets
- Dataset for challenge: https://www.kaggle.com/google/google-landmarks-dataset
- Bigger dataset: https://github.com/cvdfoundation/google-landmark
    - Resized to 256x256: http://storage.googleapis.com/landmark-recognition-2019/compressed/train-256.tar
    - Discussion: https://www.kaggle.com/c/landmark-recognition-2019/discussion/91770#latest-530831

## Experiments
### 00 Test
Only for testing purpose.
### 01 ResNet50 trained with triplet loss

