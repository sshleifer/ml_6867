import numpy as np
from sklearn.preprocessing import OneHotEncoder
# from scipy.misc import derivative


def softmax(x):
    return np.max(x, 0)

def dsigmoid(y):
    return 1.0 - y*y


def dsoftmax(x):
    return 1 if x > 0 else 0  # TODO(SS): is this right?


def stable_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)



# TODO(SS): dsoftmax?
# TODO(SS): bias at each layer
# TODO(SS): better testing
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
        self.sk_one_hot_y = OneHotEncoder(sparse=False).fit_transform(y)
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

    def _predict_row(self, x):
        '''pass one x row through the network'''
        self.a[0] = x
        for layer in range(1, self.nh + 2):
            self.z[layer] = self.w[layer].T.dot(self.a[layer - 1])  # aggregate
            self.a[layer] = self.activate(self.z[layer])
        self.zo = self.wo.T.dot(self.a[layer])
        self.ao = self.final_activate(self.zo)
        return self.ao

    def predict_probas(self, X):
        return np.apply_along_axis(self._predict_row, 1, X)

    def fit(self, X, y):
        '''stochastically'''
        for epoch in range(self.epochs):
            i = np.random.randint(0, len(X))
            x, target = X[i], self.one_hot_y[i]
            predicted_probas = self._predict_row(x)
            loss = target - predicted_probas
            self.backprop(loss)

    def backprop(self, error, lr=1.):
        '''Propagate error back through net work and update weights'''
        #delta = np.zeros((self.nh + 2, self.n_inputs))
        # import ipdb; ipdb.set_trace()
        deltas = [error]
        #delta[-1] = np.diag(map(dsoftmax, self.zo)).dot(dsigmoid(self.ao) * error)
        self.wo += (deltas[-1] * self.ao) * lr
        for layer in range(self.nh + 1, 1):
            last_w = self.w[layer + 1] if layer < self.nh + 1 else self.wo
            new_delta = np.diag(map(dsoftmax, self.z[layer])).dot(last_w).dot(deltas[-1])
            deltas.append(new_delta)
            self.w[layer] += lr * self.a[layer + 1].dot(new_delta.T)
