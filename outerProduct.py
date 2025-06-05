"""
outer product: [[1], . [[1],[2],[3]] == [[1],[2],[3]]  ->  for x = [1,2,3], xT = [1,
                [2],                    [[2],[4],[6]]                             2,
                [3]]                    [[3],[6],[9]]                             3]

Usually outerproduct is only calculated on single vectors not matrices.

"""



import numpy as np

def outerProductVector(X):

    outer_matrix = np.zeros((len(X[0]),len(X[0])))  #dxd matrix

    for i in range(len(X)):
        for j in  range(len(X)):
            outer_matrix[i][j] = X[i]*X[j]

    return outer_matrix


#X = [[1,2],[3,4]]

def outerProductMatrix(X):
    """
    outerproduct per row : creates an outer product matrix per row for a nxn X
    :param X:
    :return:
    """
    n_rows, n_cols = X.shape
    outer_matrices = []
    for row in X:
        outer_matrix = np.zeros((n_rows,n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                outer_matrix[i][j] = row[i]*row[j]

        outer_matrices.append(outer_matrix)

    return outer_matrices


if __name__ == "__main__":
    Y = np.array([[1, 2, 3], [4, 5, 6]])
    print(Y)
    X = np.array([[1, 2], [3, 4]])
    res = outerProductMatrix(X)
    for outermatrix in res:
        print(outermatrix)


