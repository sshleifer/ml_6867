import numpy as np
from sklearn.linear_model import LogisticRegression
import unittest


from code.helpers import read_in, mnist_data
from code.logistic_regression import LogReg, l1_reg
from code.svm import SVMD, Pegasos


class TestHW2(unittest.TestCase):

    def setUp(self):
        self.X, self.y = read_in(3)


    def test_log_reg(self):
        clf = LogReg(L=0.).fit(self.X, self.y)
        clf_reg = LogReg(L=1.).fit(self.X, self.y,)
        self.assertGreater(np.sum(clf.coef_), np.sum(clf_reg.coef_))
        clf_l1 = LogReg(L=1, reg_func=l1_reg).fit(self.X, self.y)

    def test_svm(self):
        clf = SVMD(C=100.).fit(self.X, self.y)
        clf_reg = SVMD(C=1.).fit(self.X, self.y)
        self.assertGreater(np.sum(clf.coef_), np.sum(clf_reg.coef_))

    def test_pegasos(self):
        clf = Pegasos(L=1.).fit(self.X, self.y)
        clf_reg = Pegasos(L=100.).fit(self.X, self.y)
        self.assertGreater(np.sum(clf.coef_), np.sum(clf_reg.coef_))

    def test_mnist_data(self):
        bigx, bigy = mnist_data(0, 7)
        clf = LogisticRegression().fit(bigx, bigy)

