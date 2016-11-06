from code.nn import NN

import numpy as np
import pandas as pd
import unittest

df = pd.read_csv('hw3_resources/data/data_3class.csv', header=None, sep=' ')
X = df[[0, 1]].as_matrix()
y = df[2].as_matrix()
n_classes = len(np.unique(y))
nn = NN(X, y)


class TestNN(unittest.TestCase):

    def test_predict(self):
        output = nn.feedforward(X[1])
        self.assertEqual(len(output), len(np.unique(y)))

    def test_backprop(self):
        nn.fit(X, y)
        self.assertGreater(np.max(nn.w.values()), 1., 'weights didnt update happen')
        predicted_probas = nn.predict_probas(X)
        self.assertEqual(predicted_probas.shape, nn.one_hot_y.shape)
        # import ipdb; ipdb.set_trace()
