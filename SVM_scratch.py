
"""
We aim to classify data points into two classes +1 and
−1 using SVM. The goal is to find the hyperplane that maximizes the margin between the classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SYM:
    def __init__(self,learning_rate,lambda_,iterations):
        self.lr = learning_rate
        self.lambda_ = lambda_ #Regularization parameter =  inverse of regularization parameter C
        self.iterations = iterations
        self.w = None #weights
        self.b = None # biases


    def fit(self,X,y):

        n_samples, n_features = X.shape
        y = np.where(y<=0,-1,1)  #np.where(condition,if true = -1,if false = 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        #Start updating weights/ training:
        for i in range(self.iterations):
            for idx,x_i in enumerate(X):
                #soft margin consideration
                condition = y[idx] * (np.dot(x_i,self.w) + self.b)  >= 1

                if condition: #True -> only update weight for regularization
                    self.w -= self.lr * (2*self.lambda_*self.w)

                else:
                    self.w -= self.lr * (2 * self.lambda_ * self.w - (y[idx]*x_i))
                    self.b -= self.lr*y[idx]

    def predict(self,X):

        y = np.dot(X,self.w) + self.b
        return np.sign(y)

# Plot the decision boundary
def plot_decision_boundary(X, y, model):
    x1 = np.linspace(0, 6, 100)
    x2 = -(model.w[0] * x1 + model.b) / model.w[1]

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
    plt.plot(x1, x2, color='green', label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Decision Boundary')
    plt.show()



# X = np.array([[1, 2], [2, 3], [3, 3], [5, 5], [6, 6], [1, 6]])
# y = np.array([1, 1, 1, -1, -1, -1])


# Generate a small dataset
X = np.array([[2, 3], [1, 1], [2, 1], [3, 2], [4, 5], [5, 4]])
y = np.array([1, 1, 1, -1, -1, -1])  # Class labels (+1 or -1)
svm = SYM(0.01,0.01,1000)
svm.fit(X,y)
plot_decision_boundary(X, y, svm)
# Plot the data
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Training Data')
plt.show()





print(f"weights : {svm.w}")
print(f"biases: {svm.b}")

# # Test data
X_test = np.array([[2, 2], [4, 4]])
predictions = svm.predict(X_test)

print("Predictions for test data:", predictions)
