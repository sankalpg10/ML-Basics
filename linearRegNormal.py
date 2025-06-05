"""
theta = (Xt.X)-1 . Xt . y
Xt : transpose of X matrix
(Xt.X)-1 :inverse of (Xt.X)
"""

import numpy as np


def linearRegression(X,y):

    Xt = np.transpose(X)

    gram_matrix = np.dot(Xt,X)

    gram_inv = np.linalg.inv(gram_matrix)

    a = np.dot(gram_inv,Xt)

    theta = np.dot(a,y)

    return np.round(theta,decimals=4)


if __name__ == "__main__":
    print(linearRegression([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))

    print(linearRegression([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1])) #[4.0, -1.0, -0.0])
