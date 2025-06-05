import numpy as np




class TwoPassNN:
    def __init__(self,input_size,hidden_size,output_size,learning_rate):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate


        #2 layers so 2 sets of weights and biases
        self.W1 = np.random.rand(hidden_size,input_size)*0.1  #hidden_ayerxinput
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.rand(output_size,hidden_size)*0.1 #hidden_ayerxinput
        self.b2 = np.zeros(output_size)

    def reLU(self,z):
        return np.maximum(z,0)

    def forward_pass(self,x):
        self.x = x
        #layer 1
        self.z1 = self.W1@x + self.b1 #h x i * i x 1 = h x 1

        self.a1 = self.reLU(self.z1) #hx1

        #layer2
        self.z2 = self.W2@self.a1 + self.b2   #o x h * h * 1 = o x 1

        self.a2 = self.reLU(self.z2)  #o x 1


        return self.a2


    def derivative_relU(self,z):
        return (z > 0).astype(float)


    def compute_loss(self,y_true,y_pred):
        self.y_true = y_true

        MSE_loss = -0.5*(np.sum(( y_pred - self.y_true)**2)) #--== y_pred = self.a2

        return MSE_loss

    def backward_pass(self):

        #Calculting derivatives of loss w.r.t
        #a2
        da2 = (self.a2 - self.y_true)
        #z2
        dz2 = da2 * self.derivative_relU(self.z2)
        #W2
        dW2 = np.outer(dz2,self.a1)
        #b2
        db2 = dz2
        #a1
        da1 = np.dot(self.W2.T,dz2)
        #z1
        dz1 = da1 * self.derivative_relU(self.z1)
        #W1
        dW1 = np.outer(dz1,self.x)
        #b2
        db1 = dz1


        #Update the weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2


    def train(self, X, Y_true,epochs):

        for epoch in range(epochs):
            total_loss = 0
            for x_,y_true_ in zip(X,Y_true):
                y_pred_ = self.forward_pass(x_)
                total_loss += self.compute_loss(y_true_,y_pred_)
                self.backward_pass()


            if epoch%10== 0 or epoch == 0:
                print(f"avg loss till epoch {epoch} : = {total_loss/len(X)}")

            return total_loss


if __name__ == "__main__":
    # Example -- for single training step

    # n, h, k = 4, 5, 3  # Input size, hidden layer size, output size
    # nn = TwoLayerNN(n, h, k,0.01)
    #
    # # Example data
    # x = np.array([1.0, 2.0, 3.0, 4.0])  # Input
    # y_true = np.array([0, 1, 0])  # One-hot true label
    #
    # # Train on a single data point
    # loss = nn.train(x, y_true)
    # print(f"Loss after training: {loss}")


    # for multiple training steps
    X = np.array([
        [1.0, 2.0, 3.0, 4.0],  # Input 1
        [0.5, 1.5, 2.5, 3.5],  # Input 2
        [0.2, 0.4, 0.6, 0.8],  # Input 3
        [3.0, 2.0, 1.0, 0.5]  # Input 4
    ])  # Shape: (4, input_size)

    Y_true = np.array([
        [0, 1, 0],  # Label 1
        [1, 0, 0],  # Label 2
        [0, 0, 1],  # Label 3
        [0, 1, 0]  # Label 4
    ])  # Shape: (4, output_size)

    nn = TwoPassNN(4,5,3,0.5)
    final_loss = nn.train(X,Y_true,10)


