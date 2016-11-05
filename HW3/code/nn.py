import numpy as np
# from sklearn.preprocessing import OneHotEncoder


def softmax(x):
    return np.max(x, 0)

def dsigmoid(y): return 1.0 - y*y

# TODO(SS): Cross-entropy loss


class NN(object):
    '''Neural Network Implementation'''

    def __init__(self, X, y, nh=2, no=4, activation_func=softmax, epochs=10):
        '''Neural Network Implementation
        Args:
            nh: # hidden layers
            n_inputs: # nodes per layer
            no: # outputs
        '''
        n_inputs = X.shape[1]
        self.z = np.zeros((nh + 2, n_inputs))
        self.a = np.zeros((nh + 2, n_inputs))
        self.w = np.ones((nh + 2, n_inputs))
        # self.one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)
        self.one_hot_y = np.array([[1 if yval == j else 0 for j in np.unique(y)] for yval in y])

        # Output shit
        #self.output = np.zeros(no)
        self.wo = np.ones((n_inputs, no))
        self.zo = np.zeros(no)

        self.n_layers = nh + 2
        self.activate = activation_func
        self.epochs = epochs
        self.bias = np.zeros(nh)
        self.nh = nh
        self.n_inputs = n_inputs
        self.d = np.array((nh, n_inputs, nh))  # matrix of errors for each layer

    @staticmethod
    def final_activate(z):
        return np.exp(z) / np.sum(np.exp(z))

    def predict(self, x=None):
        '''pass one x row through the network'''
        self.a[0] = x
        for layer in range(1, self.nh + 2):
            self.z[layer] = self.w[layer].T.dot(self.a[layer - 1])  # aggregate
            self.a[layer] = self.activate(self.z[layer])
        self.zo = self.wo.T.dot(self.a[layer])
        return self.final_activate(self.zo)

    def fit(self, X, y):
        '''stochastically'''
        for epoch in range(self.epochs):
            i = np.random.randint(0, len(X))
            x, target = X[i], self.one_hot_y[i]
            predicted_probas = self.predict(x)
            loss = -np.sum(target * np.log(predicted_probas))
            self.backprop(loss)

    def backprop(self, error):
        :0
        raise NotImplementedError
        # pseudocode


