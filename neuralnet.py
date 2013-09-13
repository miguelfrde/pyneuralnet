
import random
import math
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


def random_matrix(rows, cols, e1, e2):
    """ Create a random numpy array with values between
        e1 and e2"""
    return np.random.uniform(e1, e2, (rows, cols))


def logistic(x):
    """Logistic function: y = 1/(1 + e^-x)"""
    return np.divide(1.0, 1 + np.exp(-x))


def dlogistic(y):
    """Derivative of logistic function. y = logistic(x)"""
    return y * (1 - y)


def tanh(x):
    """Tanh function, works with numpy arrays"""
    return np.tanh(x)


def dtanh(y):
    """Derivative of tanh function. y = tanh(x)"""
    return 1 - y ** 2


class NeuralNetwork:

    """
    Implementation of a Neural Network using only
    vector and matrix operations.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 sigmoid=logistic, dsigmoid=dlogistic):
        self.sigmoid = sigmoid
        self.dsigmoid = dsigmoid
        self.input_size = input_size + 1  # 1 for bias unit
        self.hidden_size = hidden_size + 1
        self.output_size = output_size
        self.ai = np.ones(self.input_size)
        self.ah = np.ones(self.hidden_size)
        self.ao = np.zeros(self.output_size)

        # Random initial weights
        e_init = math.sqrt(6) / math.sqrt(self.input_size + self.hidden_size)
        self.weights_ih = random_matrix(self.hidden_size - 1, self.input_size,
                                        -e_init, e_init)
        e_init = math.sqrt(6) / math.sqrt(self.hidden_size + self.output_size)
        self.weights_ho = random_matrix(self.output_size, self.hidden_size,
                                        -e_init, e_init)

    def forward(self, X, p=False):
        """Perform forward propagation"""
        m = X.shape[1]
        self.ai = np.vstack((np.ones(m), X))
        z_ih = np.dot(self.weights_ih, self.ai)
        self.ah = np.vstack((np.ones(m), self.sigmoid(z_ih)))
        self.ao = self.sigmoid(np.dot(self.weights_ho, self.ah))

    def cost_function(self, theta, X, Y):
        """
        Performs forward and backpropagation to obtain the gradient of
        the cost function, that are used to update the weights.
        theta: values of weights,
        X: matrix with input data (input_size x number_of_training_examples)
        Y: matrix with expected output data 
           (output_size x number_of_training_examples)
        Returns:
        J: cost funnction.
        gradients: partial derivatives of J with respect to weights theta.
        """
        m = X.shape[1]
        self.update_weights(theta)
        self.forward(X)
        J = - np.sum(Y * np.log(self.ao) + (1 - Y) * np.log(1 - self.ao))
        J /= float(m)
        gradient_ih = np.zeros((self.hidden_size - 1, self.input_size))
        gradient_ho = np.zeros((self.output_size, self.hidden_size))
        deltak = self.ao - Y
        deltaj = np.dot(self.weights_ho.T[1:, :], deltak)
        deltaj *= self.dsigmoid(self.ah[1:, :])
        gradient_ih = np.dot(deltaj, self.ai.T) / float(m)
        gradient_ho = np.dot(deltak, self.ah.T) / float(m)
        return J, np.append(gradient_ih, gradient_ho)

    def update_weights(self, theta):
        """
        Separate the weights in theta into weights from Input to Hidden Layer
        and weights from Hidden to Output Layer.
        """
        self.weights_ih = theta[0:(self.input_size * (self.hidden_size - 1))]
        self.weights_ih = self.weights_ih.reshape(self.hidden_size - 1,
                                                  self.input_size)
        self.weights_ho = theta[(self.input_size * (self.hidden_size - 1)):]
        self.weights_ho = self.weights_ho.reshape(self.output_size,
                                                  self.hidden_size)

    def mse(self, Y):
        """Mean Squared Error"""
        m = Y.shape[1]
        norm = sum(sum(abs(self.ao - Y) ** 2))
        return norm / (2 * m)

    def minimize(self, X, Y, lim_iters, disp=False, method='BFGS'):
        """ Use CG, BFGS or L-BFGS-B algorithms to minimize the cost
            function J."""
        return optimize.minimize(lambda t: self.cost_function(t, X, Y),
                                 x0=np.append(
                                     self.weights_ih, self.weights_ho),
                                 method=method, jac=True,
                                 options={'maxiter': lim_iters, 'disp': disp})

    def train(self, X, Y, lim_iters=1000, disp=True):
        """Train the neural network"""
        X, Y = X.T, Y.T
        start = time.time()
        result = self.minimize(X, Y, lim_iters, disp)
        end = time.time()
        if disp:
            print 'Time to train: %.4f seconds.' % (end - start)
        self.update_weights(result.x)

    def train_and_errorplot(self, X, Y, lim_iters=1000, save=True,
                            error='both'):
        """ Train and plot the error J, the Mean Squared Error or both.
        Error should be one of ['both', 'j', 'mse'] """
        original_theta = np.append(self.weights_ih, self.weights_ho)
        x, yj, ymse = list(), list(), list()
        X, Y = X.T, Y.T
        for i in range(lim_iters):
            result = self.minimize(X, Y, 1)
            self.update_weights(result.x)
            x.append(i)
            ymse.append(self.mse(Y))
            yj.append(result.fun)
            if result.success:
                break
        if error in ['both', 'j']:
            plt.plot(x, yj)
        if error in ['both', 'mse']:
            plt.plot(x, ymse)
        plt.xlabel('Iterations')
        lbl = '$J(\\theta)$' if error == 'j' else \
              'MSE'if error == 'mse' else '$J(\\theta)$ and $MSE$'
        plt.ylabel('Error ' + lbl)
        plt.show()
        if not save:
            self.update_weights(original_theta)

    def train_from_files(self, x_file, y_file,
                         lim_iters=1000, error_plot=True,
                         save=True, error='both', disp=False):
        """
        Load training data from text files.
        x_file and y_file rows should be the training values
        separated by a whitespace where each row is a value xi or yi
        """
        X = np.loadtxt(x_file)
        Y = np.loadtxt(y_file)
        if error_plot:
            self.train_and_errorplot(X, Y, lim_iters, save, error)
        else:
            self.train(X, Y, lim_iters, disp)

    def save_weights(self, file_name):
        """Save the neural network weights to a file"""
        f = open(file_name, 'w')
        f.write(str(self.input_size) + ' ')
        f.write(str(self.hidden_size) + ' ')
        f.write(str(self.output_size) + '\n')

        for w in np.append(self.weights_ih, self.weights_ho):
            f.write(str(w) + '\n')
        f.close()

    def predict(self, inputs):
        """
        Perform forward propagation. inputs is a matrix of size
        number_of_training_examples x input_size
        """
        self.forward(inputs.T, True)
        return np.argmax(self.ao, axis=0)

    def test(self, inputs, outputs, disp=True):
        """
        Perform forward propagation and obtain the accuracy.
        inputs is a matrix of size number_of_training_examples x input_size
        outputs is a matrix of size number_of_training_examples x output_size
        """
        h = self.predict(inputs)
        right_samples = np.count_nonzero(h == np.argmax(outputs, axis=1))
        accuracy = right_samples / float(outputs.shape[0])
        if disp:
            print 'Accuracy: %.2f%%' % (accuracy * 100)
        return accuracy

    def test_from_files(self, x_file, y_file, disp=True):
        inputs = np.loadtxt(x_file)
        outputs = np.loadtxt(y_file)
        h = self.predict(inputs)
        right_samples = np.count_nonzero(h == np.argmax(outputs, axis=1))
        accuracy = right_samples / float(outputs.shape[0])
        if disp:
            print 'Accuracy: %.2f%%' % (accuracy * 100)
        return accuracy


if __name__ == "__main__":
    """Example -> XOR"""
    inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    outputs = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    nn = NeuralNetwork(2, 5, 2)
    nn.test(inputs, outputs)
    nn.train_and_errorplot(inputs, outputs)
    nn.test(inputs, outputs)
