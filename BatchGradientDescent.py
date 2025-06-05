"""
Batch Gradient Descent for linear regression using MSE loss fucntion
"""

import numpy as np

# Step 1: Create a simple dataset
# y = 2x + 1 (target function)
np.random.seed(42)
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 2 * X + 1 + np.random.randn(100,1 ) * 0.1  # Add some noise

#MSE loss

def mse_loss(y_pred,y_true):

    return np.mean((y_pred - y_true)**2)


def batch_gradient_descent(learning_rate,iterations,X,y,weights= None,bias= None):
    n_samples,n_features = X.shape
    weights = np.random.randn(n_features)
    bias = np.random.rand(1)

    losses = []
    for i in range(iterations):
        y_pred = X.dot(weights) + bias
        y_pred = y_pred.reshape(-1, 1)

        gradient_w = - 2*(np.dot(X.T,(y - y_pred)))/n_samples #shape = n_features
        gradient_b = -2* np.mean(y - y_pred)

        weights -= learning_rate*(gradient_w.reshape(-1))
        bias -= learning_rate*(gradient_b)


        loss = mse_loss(y_pred,y)
        losses.append(loss)

        if i%10==0:

            print(f"loss at epoch {i} : {loss}")

    return losses, weights, bias


print(batch_gradient_descent(0.1,100,X,y))



