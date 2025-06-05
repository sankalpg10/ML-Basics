"""
PCA :

1. Standardize the dataset
2. calculate cov matrix
3. calculate eigen values and eigen vectors using cov matrix (can only be calculated on a sq matrix
4. sort eigen values in desc order
5. get top k eigen vectors based on top k eigen values
6. top k eigen vectors = PCs



"""
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
means = np.mean(X,axis = 0)
# print((X - means)/np.std(X,axis = 0))
# print(X.shape[0])


def pca(data: np.ndarray, k: int) -> np.array:
    n = data.shape[0]  # no of samples
    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    cov_X = np.dot( np.transpose(X_standardized),X_standardized) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eig(cov_X)

    eigenvalues_indices = np.argsort(eigenvalues)[::-1] #sort desc
    eigenvalues = eigenvalues[eigenvalues_indices]
    eigenvectors = eigenvectors[:,eigenvalues_indices]
    principal_components = eigenvectors[:,:k]

    # Your code here
    return np.round(principal_components, 4)

data = X
print(pca(data,2))
assert pca(data,2) == np.array([[-0.7071 , 0.7071],[ 0.7071 ,0.7071]])