# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:40:40 2020

@author: Ayesha F., Aakash B.
"""

"""
Pseudo Code for Preprocessing

Frames are labeled in format: (frame, class)
Fraems are sequential and start from the first frame
Class is a numerical value of 0, 1, or 2

First
Create a fucntion that reads the CSV files and populates a numpy array
Iterate that fucntion for each patient

Second
For each patient seprate the labeled data
Store the labeled data for all patients in a separate array

Third
Split the label data into three groups
10% of the data into testing group, 90% in another group
These must be random

Fourth
Split the group containig 90% of the data into two groups
    1. Training
    2. Validation
Research k-fold training to understand how to do this
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Function to train the kNN
def TrainNetwork(training_data, training_labels, k):
    kNN = KNeighborsClassifier()
    kNN.fit(training_data, training_labels)
    KNeighborsClassifier(n_neighbors=k)
    return kNN


# Function to test the trained model
def TestNetwork(model, testing_data, testing_labels):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    testing_data : TYPE
        DESCRIPTION.
    testing_labels : TYPE
        DESCRIPTION.

    Returns
    -------
    score : TYPE
        DESCRIPTION.

    """
    score = model.score(testing_data, testing_labels)
    return score
