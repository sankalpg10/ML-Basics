"""
Two pass NN - both forward and backward pass, using ReLU actibvation using numpy
"""



import numpy as np

class TwoLayerNN:

    def __init__(self,input_size,hidden_layer_size,output_size,learning_rate=0.1): #this is for a single data point
        self.input_size = input_size #size of input vector eg.4
        self.hidden_layer_size = hidden_layer_size #no of neurons in a hidden layer eg.5
        self.output_size = output_size #size of output vector eg. 3
        self.learning_rate = learning_rate


        self.layer1_weights = np.random.randn(hidden_layer_size,input_size)* 0.01 #*0.01 so that teh weights are small values
        # size = 5x4                                                             # and to avoid exploding activation, exploding gradient
                                                                                 #problems & to have efficient training
        self.layer1_biases = np.zeros(hidden_layer_size) #initializing baises with 0s
        #size = 5X1

        self.layer2_weights = np.random.randn(output_size,hidden_layer_size) * 0.01 #size = 3x5
        self.layer2_biases = np.zeros(output_size) #size =3x1

    def reLU(self,z):
        return np.maximum(z,0)

    def forward_pass(self,x):
        print(f" in forward pass ------")

        self.x = x #input

        #layer 1 linear txn

        self.z1 = np.dot(self.layer1_weights,x) + self.layer1_biases #size = (5x4)*(4x1) = 5x1 + 5x1(biases) = 5x1
        #applying reLU activation
        self.a1 = self.reLU(self.z1) #size = 5x1

        # layer 2 linear txn

        self.z2 = np.dot(self.layer2_weights,self.a1) + self.layer2_biases #size = (3x5)*(5x1) = 3x1 + 3x1(biases) = 3x1

        self.a2 = self.reLU(self.z2)  # size = 3x1

        return self.a2

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def compute_loss(self,y_true,y_pred):
        """
        Cant use cross entropy because it is not compatible with reLU, therefore we will use mSE
        """

        self.y_true = y_true
        #cross entropy
        # cross_entropy = -np.sum(self.y_true - np.log(y_pred)) # "-" ensures loss is a positive value

        #MSE
        loss =0.5*-np.sum((y_pred - y_true)**2)
        return loss

    def backward_pass(self):
        print(f"in backward pass ------ \n ")
        #---- gradients for layer 2
        #gradient of loss w.r.t a2
        da2 = self.a2 - self.y_true  #size = 3x1 (output_size x 1) because a2 size is 3x1

        #gradient of loss w.r.t z2
        dz2 = da2* self.relu_derivative(self.z2) #size 3x1 (output_size x 1) because z2 size is 3x1 and relu derivative is applied element wise

        #gradient of loss w.r.t W2
        dW2 = np.outer(dz2,self.a1)  #outerproduct : because we want to capture hwo each weight in W2 contributes to teh loss
                                 # dz2 size = 3x1, a1 size = 5x1 , therefore outerproduct = dW2 size = output_size x hidden_layer_size = 3x5
                                    #which is the size of layer2_weights i.e. W2
        # gradient of loss w.r.t b2
        db2 = dz2 #size = 3x1 = outputsize x 1

        #---gradients for layer 1
        # gradient of loss w.r.t a1,
        da1 = np.dot(self.layer2_weights.T,dz2)  #(5x3)  x (3x1) = size = 4x1 = hidden_layer_size x 1

        # gradient of loss w.r.t z1
        dz1 = da1 * self.relu_derivative(self.z1) #hidden_layer_size x 1 = 5x1

        #gradient of loss w.r.t W1
        dW1 = np.outer(dz1,self.x) #hidden_layer_size x input_size

        # gradient of loss w.r.t b1
        db1 = dz1 #size = 5x1 = outputsize x 1

        print(f"updating weights and biases \n")
        #updating the parameters
        self.layer1_weights -= self.learning_rate*dW1
        print(f"layer1 weights : {self.layer1_weights}")
        self.layer1_biases -= self.learning_rate*db1
        print(f"layer1 biases : {self.layer1_biases}")
        self.layer2_weights -= self.learning_rate*dW2
        print(f"layer2 weights : {self.layer2_weights}")
        self.layer2_biases -= self.learning_rate*db2
        print(f"layer2 biases : {self.layer2_biases}")


    def train(self,x,y_true):

        y_pred = self.forward_pass(x)
        print(f"y_pred : {y_pred}")
        loss_ = self.compute_loss(y_true,y_pred)
        print(f"loss : {loss_}")
        self.backward_pass()

        return loss_

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

    epochs = 20

    input_size, hidden_size, output_size = 4, 5, 3
    nn = TwoLayerNN(input_size, hidden_size, output_size, learning_rate=0.5)

    for epoch in range(epochs):
        total_loss = 0
        for x, y_true in zip(X, Y_true):
            print(f" ------- epoch : {epoch}------- \n")
            loss = nn.train(x, y_true)  # Perform a single training step
            total_loss += loss

        # Print the average loss for the epoch
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n ------------------Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(X)}")


    x_test = [2.5,0.75,2.75,1.15]
    y_ = nn.forward_pass(x_test)
    print(y_)
    y_ = np.argmax(y_)
    print(y_)