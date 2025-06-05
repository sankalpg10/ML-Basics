"""
Mini Batch Gradient Descent for linear regression using MSE loss fucntion
"""

import numpy as np

# Step 1: Create a simple dataset
# y = 2x + 1 (target function)
np.random.seed(42)
X = np.random.rand(100, 1)  # 100 samples, 1 feature
y = 2 * X + 1 + np.random.randn(100,1 ) * 0.1  # Add some noise

#MSE loss

def mse_loss(y_pred,y_true):

    return np.mean((y_true - y_pred)**2)


def miniBGD(X,y,lr,iterations,batch_size):

    n_samples,n_features = X.shape
    #initialize weights and biases
    weights = np.zeros(n_features) #zeros
    bias = np.random.rand(1) #random int
    losses = []
    for epoch in range(iterations):

        #shuffle the data for each epoch

        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0,n_samples,batch_size):

            X_curr = X_shuffled[i:i+batch_size]
            y_curr = y_shuffled[i:i+batch_size]

            y_pred_batch = X_curr.dot(weights) + bias
            y_pred_batch = y_pred_batch.reshape(-1,1)

            #calculate gradients using minibatch data
            gradient_w = -2 * (np.dot(X_curr.T,(y_pred_batch - y_curr)))/batch_size
            gradient_b = -2 * np.mean(y_pred_batch - y_curr)

            #update weights
            weights -= lr*(gradient_w.reshape(-1)) #shape is n_featuresx1
            bias -= lr*(gradient_b)


            loss = mse_loss(y_pred_batch,y_curr)

        y_pred_all = X.dot(weights) + bias
        loss = mse_loss(y_pred_all.reshape(-1,1),y)
        if epoch % 10 == 0:
            print(f"Loss at epoch {epoch}: {loss}")
        losses.append(loss)

    return losses, weights,bias


# print(miniBGD(X,y,0.01,100,20))

np.random.seed(42)
X_ = np.random.rand(100, 2)  # 100 samples, 2 features
y = 2 * X_[:, 0] + 3 * X_[:, 1] + 1 + np.random.randn(100) * 0.1  # No need for (100, 1)
y_ = y.reshape(-1, 1)

print(miniBGD(X_,y_,0.01,100,20))

