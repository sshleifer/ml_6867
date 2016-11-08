from code.nn import NN
from code.hw2 import Dataset
from sklearn.linear_model import LogisticRegression


def create_nn(d, **kwargs):
    nn = NN(d.xtr, d.ytr, epochs=10000)
    nn.data = d
    return nn.fit(validate=True)


def lin_score(d):
    return (LogisticRegression()
            .fit(d.xtr, d.ytr)
            .score(d.xv, d.yv))

def do_prob_4():
    d1 = Dataset(1)
    d2 = Dataset(2)
    d3 = Dataset(3)
    d4 = Dataset(4)
    ds = [d1,d2,d3,d4]
    nns = map(create_nn, ds)
    print([nn.score_validation() for nn in nns])
    return nns
