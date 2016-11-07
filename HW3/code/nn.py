import numpy as np
np.random.seed(4)

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
        ret_val = np.isnan(struct.values()).any()
    else:
        ret_val = np.isnan(struct).any()
    if ret_val:
        raise AssertionError(struct)
        #import ipdb; ipdb.set_trace()


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
        self.X = X
        self.y = y
        self.one_hot_y = np.array([[1 if yval == j else 0 for j in np.unique(y)] for yval in y])
        self.activate = activation_func
        n_inputs = X.shape[1]
        self.epochs = epochs
        self.nh = nh
        self.L = nh + 1
        self.n_inputs = n_inputs
        self.n_hidden_nodes = hidden_nodes
        self.n_outputs = len(np.unique(y))
        self.z = np.zeros((self.L, hidden_nodes))
        self.b = np.zeros((self.L, hidden_nodes))
        self.a = np.zeros((self.L, hidden_nodes))

        self.w = {}
        #self.w[0] = np.ones(*
        for layer in range(1, self.L):
            if layer == 1:
                prev_inputs = n_inputs
            else:
                prev_inputs = len(self.a[layer - 1])

            #self.w[layer] = np.ones((prev_inputs, len(self.z[layer])))
            self.w[layer] = np.random.normal(scale=.5, size=(prev_inputs, len(self.z[layer])))
        assert len(self.w) == nh, 'more hidden weights than hidden layers'

        # Output shit
        self.wo = np.ones((hidden_nodes, self.n_outputs))
        self.bo = np.zeros(self.n_outputs)
        self.zo = np.zeros(self.n_outputs)

        self.base_loss = self.score(self.X, self.one_hot_y)

    @staticmethod
    def final_activate(z):
        return stable_softmax(z) #jlksjnp.exp(z) / np.sum(np.exp(z))

    def feedforward(self, x):
        '''pass one x row through the network'''
        self.a[0] = x
        for layer in range(1, self.L):
            assert self.w[layer].shape == (self.n_hidden_nodes, self.n_hidden_nodes)
            self.z[layer] = self.w[layer].T.dot(self.a[layer - 1]) + self.b[layer] # aggregate
            self.a[layer] = self.activate(self.z[layer])
        self.zo = self.wo.T.dot(self.a[layer]) + self.bo
        self.ao = self.final_activate(self.zo)
        return self.ao

    def _derivable_loss(self, x, y, fake_layer, layer_number):
        '''For debugging gradients'''
        self.a[0] = x
        wo = self.wo #fake_layer.reshape(2, 3)
        for layer in range(1, self.L):
            if layer == layer_number:
                self.w[layer] = fake_layer.reshape(self.n_hidden_nodes, self.n_hidden_nodes)
            assert self.w[layer].shape == (self.n_hidden_nodes, self.n_hidden_nodes)
            self.z[layer] = self.w[layer].T.dot(self.a[layer - 1]) + self.b[layer]# aggregate
            self.a[layer] = self.activate(self.z[layer])
        probas = self.final_activate(wo.T.dot(self.a[layer]) + self.bo)
        #probas = y
        return -np.sum(np.log(probas) * y)   # To be minimized

    def predict_probas(self, X):
        return np.apply_along_axis(self.feedforward, 1, X)

    def fit(self, X, y):
        '''stochastically'''
        for epoch in range(1, self.epochs):
            learning_rate = 1. / np.sqrt(epoch)
            i = np.random.randint(0, len(X))
            x, target = X[i], self.one_hot_y[i]
            predicted_probas = self.feedforward(x)
            assert_no_nans(predicted_probas)
            loss = predicted_probas - target
            self.backprop(loss, lr=learning_rate)
            if epoch % 100 == 0:
                print 'EPOCH: {}, loss = {}'.format(epoch, loss)

    def score(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.one_hot_y
        predicted_probas = self.predict_probas(X)
        cross_entropy_loss = -np.sum(np.log(predicted_probas) * y)  # 0 is perfect
        return cross_entropy_loss

    def accuracy(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.y
        yhat = np.argmax(self.predict_probas(X), axis=1)
        # import ipdb; ipdb.set_trace()
        accuracy = (yhat == y).mean()
        return accuracy

    def _updates(self, last_a, delta):
        a_reshape = last_a.reshape(last_a.shape[0], 1)
        err_reshape = delta.reshape(delta.shape[0], 1).T
        return a_reshape.dot(err_reshape)

    def backprop(self, error, lr=.1):
        '''Propagate error back through net work and update weights'''
        # delta[-1] = np.diag(map(dsoftmax, self.zo)).dot(dsigmoid(self.ao) * error)
        self.bo = self.bo - (error * lr)
        #a_reshape = self.a[self.L - 1].reshape(self.wo.shape[0], 1)
        #err_reshape = error.reshape(self.wo.shape[1], 1).T
        updates = self._updates(self.a[self.L - 1], error)
        # a_reshape.dot(err_reshape)
        assert updates.shape == self.wo.shape
        self.wo = self.wo - (updates * lr)
        #return updates * lr

        deltas = [error]
        for layer in reversed(range(1, self.L)):
            last_w = self.w.get(layer + 1, self.wo)  # next layer is output if not hidden
            new_delta = np.diag(map(dsoftmax, self.z[layer])).dot(last_w).dot(deltas[-1])
            #print 'Delta:', new_delta
            self.b[layer] = self.b[layer] - (new_delta * lr)

            # above is 2 element array
            assert new_delta.shape == (self.n_hidden_nodes,)
            updates = self._updates(self.a[layer - 1], new_delta) * lr
            #updates = self.a[layer - 1].dot(new_delta.T) * lr# should be of shape W
            if np.max(np.abs(updates)) >= 1.:
                #import ipdb; ipdb.set_trace()
                raise ValueError('Updates large: {}'.format(updates))
                #continue
            assert updates.shape == self.w[layer].shape
            assert_no_nans(updates)
            self.w[layer] = self.w[layer] - (updates * lr)
            deltas.append(new_delta)
        return deltas
