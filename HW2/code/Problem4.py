import pylab as pl
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import linear_model
import csv
import math as m
from logistic_regression import LogReg, gradient_descent,l1_reg, l2_reg
from svm import SVMD, plotDecisionBoundary, gaussian_kernel
from helpers import read_mnist, mnist_data

# functions
def process_data(name):
    data = pd.read_csv(name, header=None, delimiter=' ')
    data = data.as_matrix()
    X_val = data[:, [0, 1]]
    Y_val = data[:, -1]
    return X_val, Y_val

# constants
C_param     = [0.4, 1, 2.5]
lasso_param = [0.01, 0.04, 0.16]
gamma_param = [0.01, 0.02, 0.04]
tr = 1
fl = 7
x1 = read_mnist(tr)

# train
x1_train = x1[:200,:]
y1_train = np.ones(200)
x2 = read_mnist(fl)
x2_train = x2[:200,:]
y2_train = -np.ones(200)
x_train  = np.vstack([x1_train, x2_train])
y_train  = np.concatenate([y1_train, y2_train])

#validate
x1_valid = x1[200:350,:]
y1_valid = np.ones(150)
x2 = read_mnist(fl)
x2_valid = x2[200:350,:]
y2_valid = -np.ones(150)
x_valid  = np.vstack([x1_valid, x2_valid])
y_valid  = np.concatenate([y1_valid, y2_valid])

#test
x1_test = x1[350:500,:]
y1_test = np.ones(150)
x2 = read_mnist(fl)
x2_test = x2[350:500,:]
y2_test = -np.ones(150)
x_test  = np.vstack([x1_test, x2_test])
y_test  = np.concatenate([y1_test, y2_test])

#Linear model with lasso
# for i in range(0,len(lasso_param)):
#     lbda = lasso_param[i]
#     s = linear_model.LogisticRegression(penalty='l1', C=1.0 / lbda)
#     s.fit(x_train, y_train)
#     train_prediction = s.predict(x_train)
#     incorrect_train_prediction = (y_train != train_prediction)
#     train_error_rate = np.sum(incorrect_train_prediction) / len(y_train)
#     validation_prediction = s.predict(x_valid)
#     incorrect_validation_prediction = (y_valid != validation_prediction)
#     validation_error_rate = np.sum(incorrect_validation_prediction) / len(y_valid)
#     test_prediction = s.predict(x_test)
#     incorrect_test_prediction = (y_test != test_prediction)
#     test_error_rate = np.sum(incorrect_test_prediction) / len(y_test)
#     print("Lambda ", lbda)
#     print("Training Error ", train_error_rate)
#     print("Valid Error ", validation_error_rate)
#     print("Test Error ", test_error_rate)

#SVM model
for i in range(0,len(gamma_param)):
    gam = gamma_param[i]
    s = SVMD(method='qp', kernel=gaussian_kernel(gam))
    s.fit(x_train, y_train)
    print(s.predict(x_train[1]))
    weight = s.weight
    train_prediction = np.apply_along_axis(s.predict,1, x_train)
    incorrect_train_prediction = (y_train != train_prediction)
    train_error_rate = np.sum(incorrect_train_prediction) / len(y_train)
    validation_prediction = s.predict(x_valid)
    incorrect_validation_prediction = (y_valid != validation_prediction)
    validation_error_rate = np.sum(incorrect_validation_prediction) / len(y_valid)
    test_prediction = s.predict(x_test)
    incorrect_test_prediction = (y_test != test_prediction)
    test_error_rate = np.sum(incorrect_test_prediction) / len(y_test)
    print("Gamma ", gam)
    print("Training Error ", train_error_rate)
    print("Valid Error ", validation_error_rate)
    print("Test Error ", test_error_rate)

#x1_validate = x[:]
#print(x1_train)
