import pylab as pl
import numpy as np
import pandas as pd
import csv
import math as m
from svm import SVMD, plotDecisionBoundary, gaussian_kernel

# functions
def process_data(name):
    data = pd.read_csv(name, header=None, delimiter=' ')
    data = data.as_matrix()
    X_val = data[:, [0, 1]]
    Y_val = data[:, -1]
    return X_val, Y_val

#constants
valid_data_sets = ['data/data1_validate.csv',
                   'data/data2_validate.csv',
                   'data/data3_validate.csv',
                   'data/data4_validate.csv']
train_data_sets = ['data/data1_train.csv',
                   'data/data2_train.csv',
                   'data/data3_train.csv',
                   'data/data4_train.csv']

C_param      = [0.01,0.1, 1.0, 10.0, 100.0]
lbda_param   = [0.01, .1, 1.]
p3_gam_param = [4, 2, 1, 0.5, 0.25]
# parameters of gamma
gamma_param = []
for i in range(-12, 2):
    gamma_param.append(m.pow(2, i))
project_part = 4

if project_part ==2 :
    # print results for PB2
    PB2_results   = []
    for i in range(0, len(train_data_sets)):
        x_train, y_train = process_data(train_data_sets[i])
        x_validation, y_validation = process_data(valid_data_sets[i])
        for j in range(0, len(C_param)):
            # linear kernel results
            c = C_param[j]
            s = SVMD(C=c)
            s.fit(x_train, y_train)
            train_prediction                    = np.apply_along_axis(s.predict,1, x_train)
            incorrect_train_prediction          = (y_train!=train_prediction)
            train_error_rate                    = np.sum(incorrect_train_prediction) / len(y_train)
            num_support_vectors = len(s.svAlpha)
            num_support_vectors_at_margin = len(s.svAlpha[s.svAlpha + 0.01 > c])
            PB2_results.append([i+1,
                                c,
                                "Linear kernel",
                                s.margin,
                                num_support_vectors,
                                num_support_vectors_at_margin,
                                train_error_rate])
            # Gaussian RBF results
            for l in range(0,len(lbda_param)):
                lbda    = lbda_param[l] # the lambda in the Gaussian RBF
                s       = SVMD(C=c, kernel=gaussian_kernel(gamma=lbda))
                s.fit(x_train, y_train)
                num_support_vectors = len(s.svAlpha)
                num_support_vectors_at_margin = len(s.svAlpha[s.svAlpha + 0.01 > c])
                train_prediction = np.apply_along_axis(s.predict, 1, x_train)
                incorrect_train_prediction = (y_train != train_prediction)
                train_error_rate = np.sum(incorrect_train_prediction) / len(y_train)
                PB2_results.append([i + 1,
                                    c,
                                    "Gaussian RBF kernel, gamma=" +str(lbda),
                                    s.margin,
                                    num_support_vectors,
                                    num_support_vectors_at_margin,
                                    train_error_rate])
            # create vector of predictions and errors
            # train_prediction                    = np.apply_along_axis(s.predict,1, x_train)
            # incorrect_train_prediction          = (y_train!=train_prediction)
            # train_error_rate                    = np.sum(incorrect_train_prediction) / len(y_train)
            # validation_prediction               = np.apply_along_axis(s.predict,1, x_validation)
            # incorrect_validation_prediction     = (y_validation!=validation_prediction)
            # validation_error_rate               = np.sum(incorrect_validation_prediction) / len(y_validation)

            # store print results
            # printing is of course unnecessary
            # print("Dataset", i)
            # print("C:", c)
            # print("w:", s.weight, " b:", s.bias)
            # print("Training error rate: ",train_error_rate)
            # print("Validation error rate: ",validation_error_rate)
            # print("Margin:", s.margin)
    PB2_results = np.array(PB2_results)
    df = pd.DataFrame({'Dataset': PB2_results[:,0],
                       'C':PB2_results[:,1],
                       'Model':PB2_results[:,2],
                       'Margin':PB2_results[:,3],
                       'NumSupportVectors':PB2_results[:, 4],
                       'NumSupportVectorsAtMargin':PB2_results[:,5],
                       'TrainingErrorRate': PB2_results[:,6]})
    print(df)
    df.to_csv('data/PB2Data.csv')

if project_part ==3:
    # PB3 part 2
    PB3_p2_dat = []
    i=0
    x_train, y_train = process_data(train_data_sets[i])
    for j in range (0, len(gamma_param)):
        print(j)
        gam = gamma_param[j]
        s = SVMD(method= 'pegasos', L = gam)
        s.fit(x_train, y_train)
        PB3_p2_dat.append([gam, s.margin])
    # put in CSV
    PB3_p2_dat =np.array(PB3_p2_dat)
    print(PB3_p2_dat)
    df = pd.DataFrame({'Lambda': PB3_p2_dat[:,0],
                       'Margin': PB3_p2_dat[:,1]})
    df.to_csv('data/PB3p2Data.csv')
if project_part ==4:
    # PB3 part 4
    PB3_p4_dat = []
    i=3
    x_train, y_train = process_data(train_data_sets[i])
    #len(p3_gam_param)
    for j in range (0, len(p3_gam_param)):
        print(j)
        gam = p3_gam_param[j]
        s = SVMD(method= 'pegasos', L = 0.02, kernel=gaussian_kernel(gam))
        s.fit(x_train, y_train)
        alphas = s.alpha
        num_support_vectors = len(alphas[alphas*alphas > 0.01])
        PB3_p4_dat.append([gam, num_support_vectors])
        print(j)
        plotDecisionBoundary(x_train, y_train, s.predict, [-1.0, 1.0], title = 'Decision Boundary')
        pl.show()
    # put in CSV
    PB3_p4_dat =np.array(PB3_p4_dat)
    print(PB3_p4_dat)
    df = pd.DataFrame({'Gamma': PB3_p4_dat[:,0],
                       'NumSupportVectors': PB3_p4_dat[:,1]})
    df.to_csv('data/PB3p4Data.csv')
#Test
# for j in range(0, len(C_param)):
#  print(C_param[j])
#  data.append([j, 2*j, "Gaussian RBF, gamma =" + str(j)])
# print(data)
# out = csv.writer(open('data/test.csv',"w"),
#                  delimiter=',',
#                  quoting=csv.QUOTE_ALL,
#                  fieldnames=["A" ,"B", "C"])
# out.writerow(data)
# raise()
# data = np.array(data)
# df = pd.DataFrame({'A': data[:,0], 'B':data[:,1], 'C':data[:,2]})
# print(df)
# df.to_csv('data/test.csv')
# raise()
# read test data
# X = np.array([[2,2],
#               [2,3],
#               [0, -1],
#               [-3, -2]], dtype=float)
# Y = np.array([1, 1, -1, -1],
#              dtype=float)

#run SVM for a simple test

#s1 = SVMD(method= 'pegasos', L=0.25)
#s1.fit(X, Y)
#s0 = SVMD()
#s0.fit(X, Y)

#x_train, y_train = process_data(train_data_sets[3])
#s = SVMD()
#s.fit(x_train, y_train)
#plotDecisionBoundary(x_train, y_train, s.predict, [-1.0, 1.0], title = 'Testing')
#pl.show()
#s2 = SVMD(kernel= gaussian_kernel(gamma=0.05),
#          method = 'pegasos',
#          L=0.25)
#s2.fit(x_train, y_train)
#s3 = SVMD(kernel= gaussian_kernel(gamma=0.05),
#          method = 'qp')
#s3.fit(x_train, y_train)
#print(len(x_train),len(s3.svAlpha))
#print(np.apply_along_axis(s2.predict,1, X))
#print(np.apply_along_axis(s3.predict,1, X))
#print(s2.alpha)
#print(s3.alpha)
#s4 = SVMD()
#s4.fit(x_train, y_train)
#plotDecisionBoundary(x_train, y_train, s3.predict, [-1.0, 1.0], title = 'Testing')
#pl.show()
# raise()
# sg = SVMD(kernel= gaussian_kernel(gamma=0.05))
# sg.fit(x_train, y_train)
# train_prediction                    = np.apply_along_axis(sg.predict,1, x_train)
# incorrect_train_prediction          = (y_train!=train_prediction)
# train_error_rate                    = np.sum(incorrect_train_prediction) / len(y_train)
# print("Training error rate:", train_error_rate)
# plotDecisionBoundary(x_train, y_train, sg.predict, [-1.0,0.0, 1.0], title = 'Testing')
# pl.show()
# raise()
#print(s1.alpha)
#print(s1.weight)

#x = np.array([-1,0])
#print(s1.predict(x))

# 1. Train the model on the training data
# 2. Compute the decision boundary, and the number of misclassified examples
# on the training and validation sets
# csv_results = np.asanyarray(PB2_results)
# np.savetxt('data/PB2Results.csv',
#            csv_results,
#            delimiter=",",
#            header ="Dataset,C,Model,Margin,NumSupportVectors,NumSupportVectorsAtMargin")
# plotDecisionBoundary(x_train, y_train, s.predict, [-1.0, 1.0], title = 'Testing')
#pl.show()
# Test Gaussian RBF
# run SVM
# sg = SVMD(kernel= gaussian_kernel(gamma=0.05))
# sg.fit(x_train, y_train)
# train_prediction                    = np.apply_along_axis(sg.predict,1, x_train)
# incorrect_train_prediction          = (y_train!=train_prediction)
# train_error_rate                    = np.sum(incorrect_train_prediction) / len(y_train)
# print("Training error rate:", train_error_rate)
# plotDecisionBoundary(x_train, y_train, sg.predict, [-1.0,0.0, 1.0], title = 'Testing')
# pl.show()
#plotDecisionBoundary(x_train, y_train, s.predict, [-1.0, 1.0], title = 'Testing')
#pl.show()
#print(prediction.shape)
#print(y.shape)
#print(correct_prediction)
#print(s.alpha, len(s.svAlpha))
#plotDecisionBoundary(x, y, s.predict, [-1.0, 1.0], title = 'Testing')
#pl.show()


#print(X_val[1])
#print(s1.predict(X_val[1]))
#plotDecisionBoundary(X_val, Y_val, s1.predict, [-1.0, 1.0], title = 'Testing')
#pl.show()
#print(X_val, Y_val)
#pl.plot([1,2,3], [4,5,6], 'o')
#pl.show()
