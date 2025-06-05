import numpy as np


def batch_iterator(X, y=None, batch_size=64):
    # Your code here
    n = X.shape[0]  # total datapoints to be divided in batches of 64
    batches = []
    for i in range(0, n, batch_size):

        X_batch = X[i:i + batch_size]

        y_batch = y[i:i + batch_size]
        if y is not None:
            batches.append([X_batch, y_batch])
        else:
            batches.append(X_batch)

    print(f"output: {batches}")

    return batches

X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
y = np.array([1, 2, 3, 4, 5])
batch_size = 2
batch_iterator(X, y, batch_size)


