"""
Write a Python function to generate random subsets of a given dataset. The function should take in a 2D numpy array X,
 a 1D numpy array y, an integer n_subsets, and a boolean replacements. It should return a list of n_subsets random
 subsets of the dataset, where each subset is a tuple of (X_subset, y_subset). If replacements is True, the subsets
  should be created with replacements; otherwise, without replacements.
"""
import numpy as np

#for this - WE HAVE TPO MAKE SURE THAT THE SUBSETS ARE OF EQUAL SIZEE!!!!


def getRandomSubsets(X,y,n,replacment,seed = 42):
    n_rows = X.shape[0]
    subset_size = int(np.ceil(n_rows/n_subsets))

    subsets = []
    # np.random.seed(seed)
    if replacment:
        for s in range(n_subsets):


            start_idx = np.random.randint(0,n)
            end_idx = start_idx + subset_size if start_idx + subset_size < n_rows else n_rows

            x_subset = X[start_idx:end_idx].tolist()
            y_subset = y[start_idx:end_idx].tolist()

            subsets.append([x_subset,y_subset])

    else: #without replacement
        i = 0
        for s in range(n_subsets):

            start_idx = i
            end_idx = start_idx + subset_size if start_idx + subset_size < n_rows else n_rows
            i = end_idx
            x_subset = X[start_idx:end_idx].tolist()
            y_subset = y[start_idx:end_idx].tolist()

            subsets.append([x_subset, y_subset])

    np.random.shuffle(subsets)
    return subsets





X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
y = np.array([1, 2, 3, 4, 5])
n_subsets = 3
replacements = False
print(getRandomSubsets(X, y, n_subsets, replacements))


# for x,y in zip(X,y):
#     print(f"x: {x} -- y: {y}")