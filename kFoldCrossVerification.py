"""
Write a Python function that performs k-fold cross-validation data splitting from scratch.
The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column
represents a feature) and an integer k representing the number of folds. The function should split the dataset
 into k parts, systematically use one part as the test set and the remaining as the training set, and return a
  list where each element is a tuple containing the training set and test set for each fold.
"""
from pydoc import splitdoc

import numpy as np

def cross_validation_split(data: np.ndarray, k: int, seed=42):

    n_rows, n_cols = data.shape

    #randomy shuffle the data


    #no of rows to get in each fold
    split_size = n_rows//k
    folds = []
    i = 0
    np.random.seed(seed)
    np.random.shuffle(data)

    for i in range(k):
        test_start = i*split_size
        test_end = test_start + split_size if i < k-1 else n_rows

        test_set = data[test_start:test_end]
        train_set = np.concatenate((data[:test_start],data[test_end:]))


        fold = [train_set,test_set]
        folds.append(fold)

    return folds

data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
k = 5
print(cross_validation_split(data,k=2,seed=42))
