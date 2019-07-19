'''
Calculating random baseline for train, validation and test set.
Needs:
    - labels = landmark_id
    - random predictions
    - random confidence
Output:
    - accuracy
    - GAP score
'''

import os
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def gap_score(pred, conf, true, return_x=False):
    """ https://www.kaggle.com/davidthaler/gap-metric
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition.
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    """
    x = pd.DataFrame({"pred": pred, "conf": conf, "true": true})
    x.sort_values("conf", ascending=False, inplace=True, na_position="last")
    x["correct"] = (x.true == x.pred).astype(int)
    x["prec_k"] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x["term"] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap


def random_prediction(label, data_size):
    # create random labels from a list of landmark_id
    random_pred = np.random.choice(label, size=data_size)

    # create random confidence with uniform distribution
    random_conf = np.random.uniform(size=data_size)

    return random_pred, random_conf

def concat(pred, conf):
    concat_pred_conf = str(pred) + ' ' + str(conf)
    return concat_pred_conf


def create_submission(id, prediction, confidence_score):
    sub = pd.DataFrame(columns=['id', 'landmarks'])
    sub['id'] = id

    landmarks = []
    for pred, conf in tqdm(zip(prediction, confidence_score)):
        landmarks.append(concat(pred, conf))
    sub['landmarks'] = landmarks

    # write to a csv file
    data_path = os.path.dirname(os.path.realpath('')) + '/data/'
    filename = data_path + 'random_baseline_submission.csv'
    sub.to_csv(filename, index=False)




########### Preparing data
# load data
data_path = os.path.dirname(os.path.realpath('')) + '/data/google-landmarks-dataset/'
train = pd.read_csv(data_path + 'train.csv', sep=',')
test = pd.read_csv(data_path + 'test.csv', sep=',')

# get unique labels
train_labels = train['landmark_id'].unique()

# split data in train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(train, train['landmark_id'], test_size=0.25)

# data size
train_size = len(X_train)
valid_size = len(X_valid)
test_size = len(test)


########### Prediction
train_pred, train_conf = random_prediction(train_labels, train_size)
valid_pred, valid_conf = random_prediction(train_labels, valid_size)
test_pred, test_conf = random_prediction(train_labels, test_size)


########### Metrics
# calculate accuracy
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

# calculate GAP score
train_gap = gap_score(train_pred, train_conf, y_train)
valid_gap = gap_score(valid_pred, valid_conf, y_valid)


########### test submission
create_submission(test['id'], test_pred, test_conf)


########### printing results
print('-' * 50)
print('Random baseline for training dataset: ')
print('Accuracy: ', train_acc)
print('GAP:', train_gap)

print('-' * 50)
print('Random baseline for validation dataset: ')
print('Accuracy: ', valid_acc)
print('GAP:', valid_gap)
