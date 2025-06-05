from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Data
X = np.array([[1, 2], [2, 3], [3, 3], [5, 5], [6, 6], [1, 6]])
y = np.array([1, 1, 1, -1, -1, -1])

# Train soft margin SVM
clf = SVC(C=1.0, kernel='linear')
clf.fit(X, y)

# Plot decision boundary
def plot_svm_boundary(clf, X, y):
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')

    # Plot decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    plt.legend()
    plt.show()

plot_svm_boundary(clf, X, y)
