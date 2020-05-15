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
from sklearn.model_selection import RepeatedKFold


# Function to train the kNN
def TrainNetwork(training_data, training_labels, k):
    """
    Train a kNN to the training data wiht various number of neighbors.

    Parameters
    ----------
    training_data : ndarray
        Array of the data that needs to be classified in the form
        [samples][features].
    training_labels : ndarray
        Array containing the classifier labels for the training data.
    k : intger
        The number of nearest neighbors that should be considered by the model.

    Returns
    -------
    kNN : sklearn.neighbors.KNeighborsClassfier
        Trained kNN model.

    """
    kNN = KNeighborsClassifier()
    kNN.fit(training_data, training_labels)
    KNeighborsClassifier(n_neighbors=k)
    return kNN


# Function to test the trained model
def TestNetwork(model, testing_data, testing_labels):
    """
    Use a pretrained kNN to test against labeled data.

    Parameters
    ----------
    model : sklearn.neighbors.KNeighborsClassfier
        Classfier object that has been pretrained using testing data.
    testing_data : ndarray
        Array of data that needs to be tested in form [samples][features].
    testing_labels : ndarray
        Array containig the actual labels for the test data.

    Returns
    -------
    score : float
        Mean of the accuracy of the test classification.

    """
    score = model.score(testing_data, testing_labels)
    return score


# Determine Factors of number of data samples to split it evenly
def Factorize(num):
    """
    Return all the factors of a number excluding 1.

    Parameters
    ----------
    num : integer
        Whole integer number for which factor must be found.

    Returns
    -------
    factors : list
        List containing interger factors of the input number.

    """
    factors = []
    for i in range(1, num+1):
        if num % i == 0:
            factors.append(i)
    return factors


# Function to split data in k-folds
# Splits and repeats should be factors of the number of samples
# Sample Data is all the data, sample labels are all the labels
def PerformKSplit(sample_data, splits, repeats):
    """
    Take input data and splits it into training and validation sets.

    The function performs a repeated k-fold split that is randomized. Trainig
    data contains n-1 groups of samples, and validation data contains 1 group
    of samples. The number of samples in the group depends on the value of
    splits. The number of times this process should be done is dictated by
    value of repeats.

    Parameters
    ----------
    sample_data : ndarray
        Input data in the form [samples][features].
    splits : integer
        Number of splits that should be made.
    repeats : integer
        Number of times the process should be repeated.

    Returns
    -------
    q : ndarray
        Contains the index values for trainig data set for each iteration. The
        form of the array is [iteration][indices].
    w : ndarray
        Contains the index values for validation data for each iteration. The
        form of the array is [iteration][indices]

    """
    rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats,
                        random_state=2652124)
    index = 0
    # Assign q, number of rows is the the number of splits times the repeats
    # The number of columns is total number of samples minus on group
    # sample_labels.shape[0] provides the total number of samples
    # dividing the total number of samples by splits gives the number of
    # samples in each group
    # Training set 'q' will have n-1 groups
    # Testing set 'w' will have 1 group
    rows = int(splits*repeats)
    test_cols = int(sample_data.shape[0]/splits)
    train_cols = int(sample_data.shape[0]-test_cols)
    q = np.zeros((rows, train_cols))
    w = np.zeros((rows, test_cols))
    for train_index, test_index in rkf.split(sample_data):
        q[index] = train_index
        w[index] = test_index
        index += 1
    return q, w


# Function to perform k-fold cross validaton
def PerformKFold(sample_data, sample_labels, splits, repeats, neighbors):
    """
    Perform kNN classification using k-fold cross validation.

    Parameters
    ----------
    sample_data : ndarray
        Input data in form of [samples][features].
    sample_labels : ndarray
        Desired classifier output in form [samples][label].
    splits : integer
        Number of splits that should be made in the sample data.
    repeats : integer
        Number of times the splits should be repeated.
    neighbors : int
        Number of neighbours that should be taken into consideration for kNN
        classification.

    Returns
    -------
    scores : ndarray
        Array of average score values for each iteration.

    """
    # First need to split the data into k repititions
    train_index, val_index = PerformKFold(sample_data, splits, repeats)
    scores = np.zeros(val_index.shape[0])
    for iteration in range(train_index.shape[0]):
        # Train the network
        training_data = sample_data[train_index]
        validation_data = sample_data[val_index]
        training_labels = sample_labels[train_index]
        validation_labels = sample_labels[val_index]
        kNN = TrainNetwork(training_data, training_labels, neighbors)
        # Test the network
        score = TestNetwork(kNN, validation_data, validation_labels)
        scores[iteration] = score
    return scores
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    