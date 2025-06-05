"""
To calculate covariance matrix:

Method - 1:
1. centralize all feature vectors i.e. substract mean
2. calculate outer product of all vectors with themselves
3. sum all the out product matrices and divide by number of vectors

outer product: [[1], . [[1],[2],[3]] == [[1],[2],[3]]  ->  for x = [1,2,3], xT = [1,
                [2],                    [[2],[4],[6]]                             2,
                [3]]                    [[3],[6],[9]]                             3]

Method - 2:
cov matrix :
1. centralize all feature vectors i.e. substract mean
2. calculate dot product of centralized.Transpose and centralized
3. divide it by n-1



Method - 3
1. use formula: (1/(n-1))* sum of ((Xi-meani)*(Yi-meani) for i features
"""


import numpy as np

X = np.array([[1,4],[2,5],[3,6]])
Y = np.array([[1, 2, 3], [4, 5, 6]])
# Xt = np.transpose(X)
# print(X)


def getCovarianceMatrix(X):

    X_centered = X - np.mean(X, axis=1, keepdims=True)
    print(X_centered)
    n = X.shape[1]
    print(n)
    cov_matrix = np.dot(np.transpose(X_centered),X_centered,)/(n-1)

    return cov_matrix


def getCovariance(X): #[[1,2,3],[4,5,6]] #2 features 3 samples ,therefore, cov_matrix = 2x2 shape
    """
    X is a dataframe of features, we want to calculate teh matrix w/o using numoy
    """
    n_features = len(X)
    n_samples = len(X[0])

    cov_matrix = [[0 for _ in range(n_features)] for _ in range(n_features)]

    means = [sum(row)/n_samples for row in X] #vector of means for each feature

    #now we want to calculate i,j index values for the cov_matrix
    #
    for i in range(n_features):
        for j in range(n_features):

            covariance = (sum((X[i][k] - means[i])*(X[j][k] - means[j]) for k in range(n_samples)))*(1/(n_samples-1))
            cov_matrix[i][j] = cov_matrix[j][i] = covariance

    return cov_matrix


if __name__ == "__main__":
    Y = np.array([[1, 2, 3], [4, 5, 6]])
    print(Y)
    X = np.array([[1, 4], [2, 5], [3, 6]])
    print(getCovarianceMatrix(Y))
    X = [[1, 2, 3], [4, 5, 6]]
    print(getCovariance(X))
