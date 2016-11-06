import numpy as np


def relu(x):
    return np.clip(x, 0, None)

def dsigmoid(y):
    return 1.0 - y * y


def dsoftmax(x):
    return np.clip(np.sign(x), 0, 1)


def stable_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def assert_no_nans(struct):
    if isinstance(struct, dict):
        assert not np.isnan(struct.values()).any()
    else:
        assert not np.isnan(struct).any()


# TODO(SS): better testing
# TODO(SS): Cross-entropy loss


class NN(object):
    '''Neural Network Implementation'''

    def __init__(self, X, y, nh=2, hidden_nodes=2, activation_func=relu, epochs=100):
        '''Neural Network Implementation
        Args:
            nh: # hidden layers
            n_inputs: # nodes per layer
            no: # outputs
        '''
        n_inputs = X.shape[1]
        self.epochs = epochs
        self.nh = nh
        self.L = nh + 2
        self.n_inputs = n_inputs
        self.n_hidden_nodes = hidden_nodes
        self.n_outputs = len(np.unique(y))
        self.z = np.zeros((self.L, hidden_nodes))
        self.b = np.zeros((self.L, hidden_nodes))
        self.a = np.zeros((self.L, hidden_nodes))
        self.w = {}#np.ones((self.L, hidden_nodes))
        self.one_hot_y = np.array([[1 if yval == j else 0 for j in np.unique(y)] for yval in y])

        # Output shit
        self.wo = np.ones((hidden_nodes, self.n_outputs))
        self.bo = np.zeros(self.n_outputs)
        self.zo = np.zeros(self.n_outputs)

        self.activate = activation_func

    @staticmethod
    def final_activate(z):
        return np.exp(z) / np.sum(np.exp(z))

    def feedforward(self, x):
        '''pass one x row through the network'''
        self.a[0] = x
        for layer in range(1, self.L):
            if layer not in self.w:
                # self.w[layer] = np.ones((len(self.a[layer - 1]), len(self.z[layer])))
                self.w[layer] = np.random.rand(len(self.a[layer - 1]), len(self.z[layer]))
            assert self.w[layer].shape == (self.n_hidden_nodes, self.n_hidden_nodes)
            self.z[layer] = self.w[layer].T.dot(self.a[layer - 1]) + self.b[layer]# aggregate
            self.a[layer] = self.activate(self.z[layer])
        self.zo = self.wo.T.dot(self.a[layer]) + self.bo
        self.ao = self.final_activate(self.zo)
        return self.ao

    def predict_probas(self, X):
        return np.apply_along_axis(self.feedforward, 1, X)

    def fit(self, X, y):
        '''stochastically'''
        for epoch in range(self.epochs):
            i = np.random.randint(0, len(X))
            x, target = X[i], self.one_hot_y[i]
            predicted_probas = self.feedforward(x)
            assert_no_nans(predicted_probas)
            loss = target - predicted_probas
            self.backprop(loss)
            # cross_entropy_loss = -np.sum(np.log(predicted_probas) * y)
            #print cross_entropy_loss
            # self.backprop(cross_entropy_loss)

    def score(self, X, y):
        predicted_probas = self.predict_probas(X)
        cross_entropy_loss = -np.sum(np.log(predicted_probas) * y)
        return cross_entropy_loss

    def _updates(self, last_a, delta):
        a_reshape = last_a.reshape(last_a.shape[0], 1)
        err_reshape = delta.reshape(delta.shape[0], 1).T
        return a_reshape.dot(err_reshape)

    def backprop(self, error, lr=1.):
        '''Propagate error back through net work and update weights'''
        # delta[-1] = np.diag(map(dsoftmax, self.zo)).dot(dsigmoid(self.ao) * error)
        self.bo -= error * lr
        #a_reshape = self.a[self.L - 1].reshape(self.wo.shape[0], 1)
        #err_reshape = error.reshape(self.wo.shape[1], 1).T
        updates = self._updates(self.a[self.L - 1], error)
        # a_reshape.dot(err_reshape)
        assert updates.shape == self.wo.shape
        self.wo = self.wo - updates * lr

        deltas = [error]
        for layer in reversed(range(1, self.L)):
            last_w = self.w[layer + 1] if layer < self.nh + 1 else self.wo
            new_delta = np.diag(map(dsoftmax, self.z[layer])).dot(last_w).dot(deltas[-1])
            self.b[layer] = self.b[layer] - new_delta

            # above is 2 element array
            assert new_delta.shape == (self.n_hidden_nodes,)
            updates = self._updates(self.a[layer - 1], new_delta) * lr
            #updates = self.a[layer - 1].dot(new_delta.T) * lr# should be of shape W
            assert updates.shape == self.w[layer].shape
            assert_no_nans(updates)
            self.w[layer] = self.w[layer] - updates
            deltas.append(new_delta)
        return deltas
