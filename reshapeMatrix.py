import numpy as np



"""From Scratch"""
def reshapeMatrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    #Write your code here and return a python list after reshaping by using numpy's tolist() method
    m = len(a)
    n = len(a[0])

    total_elements = m*n

    elements = [ele for sublist in a for ele in sublist]
    reshaped_matrix = [[0]*new_shape[1] for _ in range(new_shape[0])]
    k =0
    if new_shape[0]*new_shape[1] == total_elements:

        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                reshaped_matrix[i][j] = elements[k]
                k+=1


    return reshaped_matrix


"""Using numpy"""


def reshapeMatrixNP(a,new_shape):

    return np.array(a).reshape(new_shape).tolist()


