import pylab as pl
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import linear_model
import csv
import math as m
from logistic_regression import LogReg, gradient_descent,l1_reg, l2_reg
from svm import SVMD, plotDecisionBoundary, gaussian_kernel

# functions
def process_data(name):
    data = pd.read_csv(name, header=None, delimiter=' ')
    data = data.as_matrix()
    X_val = data[:, [0, 1]]
    Y_val = data[:, -1]
    return X_val, Y_val
def make_number(x):
    return s.predict(x)[0]
lambda_val = [0.01, 0.1, 1., 10.]

#constants
valid_data_sets = ['data/data1_validate.csv',
                   'data/data2_validate.csv',
                   'data/data3_validate.csv',
                   'data/data4_validate.csv']
train_data_sets = ['data/data1_train.csv',
                   'data/data2_train.csv',
                   'data/data3_train.csv',
                   'data/data4_train.csv']
test_data_sets = ['data/data1_test.csv',
                   'data/data2_test.csv',
                   'data/data3_test.csv',
                   'data/data4_test.csv']

ResData =[]
for i in range(0, len(train_data_sets)):
    x_train, y_train = process_data(train_data_sets[i])
    x_validation, y_validation = process_data(valid_data_sets[i])
    x_test, y_test = process_data(test_data_sets[i])
    lbda = 0.02
    s = linear_model.LogisticRegression(penalty='l1', C=1.0/lbda)
    s.fit(x_train, y_train)
    w= s.coef_
    # create vector of predictions and errors
    train_prediction                    = s.predict(x_train)
    incorrect_train_prediction          = (y_train!=train_prediction)
    train_error_rate                    = np.sum(incorrect_train_prediction) / len(y_train)
    validation_prediction               = s.predict(x_validation)
    incorrect_validation_prediction     = (y_validation!=validation_prediction)
    validation_error_rate               = np.sum(incorrect_validation_prediction) / len(y_validation)
    test_prediction                     = s.predict(x_test)
    incorrect_test_prediction           = (y_test!=test_prediction)
    test_error_rate                     = np.sum(incorrect_test_prediction) / len(y_test)
    print("Dataset:", i+1)
    print("Loss: L1")
    print("Lambda:", lbda)
    print("Weights", w, s.intercept_)
    print("Train error:", train_error_rate)
    print("Valid error:", validation_error_rate)
    print("Test error: ", test_error_rate)
    print("---")
# for i in range(0, len(train_data_sets)):
#     x_train, y_train = process_data(train_data_sets[i])
#     x_validation, y_validation = process_data(valid_data_sets[i])
#     for j in range(0, len(lambda_val)):
#         lbda = lambda_val[j]
#         s = linear_model.LogisticRegression(penalty='l1', C=1.0/lbda)
#         s.fit(x_train, y_train)
#         w= s.coef_
#         # create vector of predictions and errors
#         train_prediction                    = s.predict(x_train)
#         incorrect_train_prediction          = (y_train!=train_prediction)
#         train_error_rate                    = np.sum(incorrect_train_prediction) / len(y_train)
#         validation_prediction               = s.predict(x_validation)
#         incorrect_validation_prediction     = (y_validation!=validation_prediction)
#         validation_error_rate               = np.sum(incorrect_validation_prediction) / len(y_validation)
#         print("Dataset:", i+1)
#         print("Loss: L1")
#         print("Lambda:", lbda)
#         print("Weights", w, s.intercept_)
#         print("Train error:", train_error_rate)
#         print("Valid error:", validation_error_rate)
#         print("---")
#     for j in range(0, len(lambda_val)):
#         lbda = lambda_val[j]
#         s = linear_model.LogisticRegression(penalty='l2', C = 1/(2*lbda))
#         s.fit(x_train, y_train)
#         w= s.coef_
#         # create vector of predictions and errors
#         train_prediction                    = s.predict(x_train)
#         incorrect_train_prediction          = (y_train!=train_prediction)
#         train_error_rate                    = np.sum(incorrect_train_prediction) / len(y_train)
#         validation_prediction               = s.predict(x_validation)
#         incorrect_validation_prediction     = (y_validation!=validation_prediction)
#         validation_error_rate               = np.sum(incorrect_validation_prediction) / len(y_validation)
#         print("Dataset:", i+1)
#         print("Loss: L2")
#         print("Lambda:", lbda)
#         print("Weights", w, s.intercept_)
#         print("Train error:", train_error_rate)
#         print("Valid error:", validation_error_rate)
# hist_vect = w1[:,[2,3,4]]
# hist_norm = []
# for i in range(0, len(hist_vect)):
#     v= hist_vect[i]
#     print(v)
#     norm = sum(v[1:,]*v[1:,])
#     hist_norm.append([i, norm])
# hist_norm = np.array(hist_norm)
# df = pd.DataFrame({'Iter': hist_norm[:, 0],
#                    'L2Norm': hist_norm[:, 1]})
# df.to_csv('data/P1HistNorm.csv')
# Model is working properly
#plotDecisionBoundary(x_train, y_train, s0.predict, [-1.0, 1.0], title='Decision Boundary, L=1')
#pl.show()

