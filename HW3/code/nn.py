from __future__ import division
import numpy as np
from code.helpers import relu, dsoftmax, stable_softmax, assert_no_nans, get_lr
np.random.seed(4)



class NN(object):
    '''Neural Network Implementation'''

    def __init__(self, X, y, nh=2, n_hidden_nodes=2, activation_func=relu, epochs=100):
        '''Neural Network Implementation
        Args:
            nh: # hidden layers
            n_inputs: # nodes per layer
            no: # outputs
        '''
        self.X = X
        self.y = y
        self.classes = sorted(np.unique(y))
        self.ymap = dict(enumerate(self.classes))
        self.one_hot_y = np.array([[1 if yval == j else 0 for j in self.classes] for yval in y])

        self.activate = activation_func
        n_inputs = X.shape[1]
        self.epochs = epochs
        self.nh = nh
        self.L = nh + 1
        self.n_inputs = n_inputs
        self.n_hidden_nodes = n_hidden_nodes
        self.n_outputs = len(np.unique(y))
        self.z = np.zeros((self.L, n_hidden_nodes))
        self.b = np.zeros((self.L, n_hidden_nodes))
        self.a = {}
        np.zeros((self.L, n_hidden_nodes))

        self.w = {}
        #self.w[0] = np.ones(*
        for layer in range(self.L):
            if layer == 0:
                self.a[layer] = np.zeros(self.n_inputs)
                continue

            self.a[layer] = np.zeros(self.n_hidden_nodes)
            prev_inputs = len(self.a[layer - 1])
            #self.w[layer] = np.ones((prev_inputs, len(self.z[layer])))
            scale = .5 / np.sqrt(self.n_hidden_nodes)
            self.w[layer] = np.random.normal(scale=scale, size=(prev_inputs, n_hidden_nodes))
        assert len(self.w) == nh, 'more hidden weights than hidden layers'

        # Output shit
        self.wo = np.ones((n_hidden_nodes, self.n_outputs))
        self.bo = np.zeros(self.n_outputs)
        self.zo = np.zeros(self.n_outputs)
        self.base_loss = self.score(self.X, self.one_hot_y)

    def revy(self, labels):
        return np.array([self.ymap.get(l) for l in labels])

    def onehot(self, y):
        return np.array([[1 if yval == j  else 0 for j in self.classes] for yval in y])

    def score_validation(self):
        return self.accuracy(self.data.xv, self.data.yv)


    @staticmethod
    def final_activate(z):
        return stable_softmax(z) #jlksjnp.exp(z) / np.sum(np.exp(z))

    def feedforward(self, x):
        '''pass one x row through the network'''
        self.a[0] = x
        for layer in range(1, self.L):
            # assert self.w[layer].shape == (self.n_hidden_nodes, self.n_hidden_nodes)
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

    def fit(self, validate=False):
        '''stochastically'''
        self.cur_epoch = 0
        last_score = 0.
        for epoch in range(1, self.epochs):
            self.cur_epoch = epoch
            #learning_rate = get_lr(epoch, k=1.)
            learning_rate = max(1. / epoch, 1. / len(self.X))
            i = np.random.randint(0, len(self.X))
            x, target = self.X[i], self.one_hot_y[i]
            predicted_probas = self.feedforward(x)
            assert_no_nans(predicted_probas)
            loss = predicted_probas - target
            self.backprop(loss, lr=learning_rate)
            if epoch % max(1e3, self.epochs / 10) == 0:
                print('EPOCH: {}, accuracy = {}'.format(epoch, self.accuracy()))
                if validate:
                    new_score = self.score_validation()
                    if new_score <= last_score or new_score == 1:
                        break
                    else:
                        last_score = new_score

        return self

    def score(self, X=None, y=None):
        if X is None:
            X = self.X
        if y is None:
            y = self.one_hot_y
        predicted_probas = self.predict_probas(X)
        cross_entropy_loss = -np.sum(np.log(predicted_probas) * y)  # 0 is perfect
        return cross_entropy_loss

    def predict(self, X=None):
        if X is None:
            X = self.X
        return self.revy(np.argmax(self.predict_probas(X), axis=1))

    def accuracy(self, X=None, y=None):
        '''How often is highest predicted proba class the actual class'''
        if y is None:
            y = self.y
        yhat = self.predict(X)
        accuracy = (yhat == y).mean()
        return accuracy

    def _updates(self, last_a, delta):
        a_reshape = last_a.reshape(last_a.shape[0], 1)
        err_reshape = delta.reshape(delta.shape[0], 1).T
        return a_reshape.dot(err_reshape)

    def backprop(self, error, lr=.1):
        '''Propagate error back through net work and update weights'''
        self.bo = self.bo - (error * lr)
        output_weight_updates = self._updates(self.a[self.L - 1], error)
        assert output_weight_updates.shape == self.wo.shape
        self.wo = self.wo - (output_weight_updates * lr)

        deltas = [error]
        for layer in reversed(range(1, self.L)):
            last_w = self.w.get(layer + 1, self.wo)  # next layer is output if not hidden
            new_delta = np.diag(map(dsoftmax, self.z[layer])).dot(last_w).dot(deltas[-1])
            assert new_delta.shape == (self.n_hidden_nodes,)
            self.b[layer] = self.b[layer] - (new_delta * lr)
            weight_updates = np.clip(self._updates(self.a[layer - 1], new_delta) * lr, -1, 1)
            assert weight_updates.shape == self.w[layer].shape
            assert_no_nans(weight_updates)
            if np.max(np.abs(weight_updates)) >= max(np.max(np.abs(self.w[layer])), 1e3):
                weight_updates = weight_updates / 1e3
                #raise ValueError('Updates large: {}, gradient explosion at epoch {}'.format(weight_updates, self.cur_epoch))

            self.w[layer] = self.w[layer] - weight_updates
            deltas.append(new_delta)
