

def multiply(mat1, mat2):
    n_rows = len(mat1)  # m
    n_cols = len(mat2[0])  # n
    iters = len(mat2)  # k

    mat = [[0 for i in range(n_cols)] for j in range(n_rows)]

    print(mat)

    for i in range(n_rows):
        for j in range(n_cols):
            mat[i][j] = sum([mat1[i][k] * mat2[k][j] for k in range(iters)])

    return mat