"""
Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X
(features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the
number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer
 to four decimal places. -0.0 is a valid result for rounding a very small number.
"""


import numpy as np

def linearRegressionGD(X,y,alpha,iters):

    m,n = X.shape
    theta = np.zeros((n,1))

    for i in range(iters):

        #calculate preds
        preds = np.dot(X,theta)

        #calculate errors
        errors = preds - y.reshape(-1,1)

        #calculate updates - gradient
        updates = (np.dot(np.transpose(X),errors))/m

        #update theta

        theta = theta - (alpha*updates)

    return theta


print(linearRegressionGD(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000))
