# [Landmark Recognition Challenge](https://www.kaggle.com/c/landmark-recognition-2019)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Preparation
### Create Environment
- Navigate into this repository
- Execute following command: `conda env create -f environment.yml`
- Activate the environment: `conda activate landmark-recognition`
### Download Datasets
- Download one or more datasets from [Datasets](#datasets) 
    - It doesn't matter where you save them
    - Don't rename the downloaded csv. They should be named `train` or `test`.
- Navigate into this repository
- Activate the environment
- Execute following command: 
    `python scripts/download_dataset.py --name={FOLDER_NAME} --csv={PATH_TO_DOWNLOADED_CSV}`
    - You have to download train and test images separately
    - Images are saved in `./data/{FOLDER_NAME}/{CSV_NAME}/{ID}.jpg`
    - For testing purpose you can also download only the first N images with passing `--num {N}`

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
Start experiments with `python main.py {ID} {FLAGS}`
Checkpoints, evaluation artifacts, logs are stored in sub directories named after the experiment and passed flags.
### 00 Test
Only for testing purpose.
### 01 ResNet50 trained with triplet loss
Crop input image into 5 sub images and extract features on each one. Uses triplet loss. Does not work yet!
