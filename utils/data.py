"""
File: data.py
Description: Contains utils for dataset stuff
"""

import numpy as np
import sklearn


class LabelEncoder:
    """ Encodes labels so that each label is in [0, n_classes) and 
        sorted by occurence
    """

    def __init__(self):
        self.label_encoder = sklearn.preprocessing.LabelEncoder()
        self.slabel2label = None
        self.label2slabel = None

    def fit_transform(self, labels):
        labels = self.label_encoder.fit_transform(labels)
        unique_labels, label_count = np.unique(labels, return_counts=True)
        # labels with respect to number of occurence
        self.slabel2label = unique_labels[np.flip(np.argsort(label_count))]
        self.label2slabel = np.argsort(self.slabel2label)
        transformed_labels = self.label2slabel[labels]
        return transformed_labels

    def inverse_transform(self, labels):
        labels = self.slabel2label[labels]
        orig_labels = self.label_encoder.inverse_transform(labels)
        return orig_labels
