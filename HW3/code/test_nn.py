from code.hw2 import Dataset
from code.nn import NN

import numpy as np
import pandas as pd
import unittest

df = pd.read_csv('hw3_resources/data/data_3class.csv', header=None, sep=' ')
X = df[[0, 1]].as_matrix()
y = df[2].as_matrix()
d = Dataset(1)

class TestNN(unittest.TestCase):

    def test_predict(self):
        nn = NN(X, y, nh=1)
        output = nn.feedforward(X[1])
        self.assertEqual(len(output), len(np.unique(y)))

    def test_backprop(self):
        nn = NN(X, y, nh=1)
        start_weights = nn.w
        base_loss = nn.score(X, nn.one_hot_y)
        nn.fit()
        movement = np.sum(nn.w[1]) - start_weights[1]
        self.assertGreater(np.sum(np.abs(movement)), 0, 'weights didnt update')
        predicted_probas = nn.predict_probas(X)
        self.assertEqual(predicted_probas.shape, nn.one_hot_y.shape)
        self.assertGreater(base_loss, nn.score(X, nn.one_hot_y),
                           'training did not reduce loss {}, was {}'.format(
                               nn.score(X, nn.one_hot_y), base_loss
                           ))

    def test_two_hidden(self):
        nn = NN(X, y, nh=2, epochs=1000)
        start_weights = nn.w
        nn = nn.fit()
        movement = np.sum(nn.w[1]) - start_weights[1]
        self.assertGreater(np.sum(np.abs(movement)), 0, 'weights didnt update')
        predicted_probas = nn.predict_probas(X)
        self.assertEqual(predicted_probas.shape, nn.one_hot_y.shape)
        self.assertGreater(nn.base_loss, nn.score(X, nn.one_hot_y),
                           'training did not reduce loss {}, was {}'.format(
                               nn.score(X, nn.one_hot_y), nn.base_loss
                           ))
        self.assertGreater(nn.accuracy(), .5)

    def test_big_hidden(self):
        nn = NN(X, y, nh=1, n_hidden_nodes=3)
        start_weights = nn.w
        base_loss = nn.score(X, nn.one_hot_y)
        nn = nn.fit()
        movement = np.sum(nn.w[1]) - start_weights[1]
        self.assertGreater(np.sum(np.abs(movement)), 0, 'weights didnt update')
        predicted_probas = nn.predict_probas(X)
        self.assertEqual(predicted_probas.shape, nn.one_hot_y.shape)
        self.assertGreater(base_loss, nn.score(X, nn.one_hot_y),
                           'training did not reduce loss {}, was {}'.format(
                               nn.score(X, nn.one_hot_y), base_loss)
                           )
        self.assertGreater(nn.accuracy(), .5)

    def test_linear_separable(self):
        nn = NN(d.xtr, d.ytr, nh=1, n_hidden_nodes=1, epochs=1000).fit()
        self.assertGreaterEqual(nn.accuracy(), .9)

    def test_validation_stopping(self):
        nn = NN(d.xtr, d.ytr, nh=1, n_hidden_nodes=1, epochs=10000)
        start_weights = nn.w
        nn.data = Dataset(1)
        nn = nn.fit(validate=True)
        movement = np.sum(nn.w[1]) - start_weights[1]
        self.assertGreater(np.sum(np.abs(movement)), 0, 'weights didnt update')
        predicted_probas = nn.predict_probas(nn.X)
        self.assertEqual(predicted_probas.shape, nn.one_hot_y.shape)
        cross_ent = nn.score(nn.X, nn.one_hot_y),
        self.assertGreater(nn.base_loss, cross_ent,
                           'training did not reduce loss {}, was {}'.format(cross_ent, nn.base_loss)
                           )
        self.assertGreater(nn.accuracy(), .5)
