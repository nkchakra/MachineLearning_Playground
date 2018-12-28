import numpy as np
import pandas as pd
import sys
import os
import csv

import matplotlib.pyplot as plt

# np array of floats
# finished


def store_model(model):
    with open('model.csv','w') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        writer.writerow(model)



def normalize(data):
    for i in range(len(data[0])):

        col_avg = np.average(data[:,i])

        data[:,i]=data[:,i]-col_avg

        rnge = max(data[:,i])-min(data[:,i])
        if(rnge < 1/sys.float_info.max):
            rnge = 1/sys.float_info.max
        data[:,i]=data[:,i]/rnge
    return data

def normalize_Y(Y):
    return (Y-np.average(Y))/(max(Y)-min(Y))
def avg_diff(data,results,model):
    diff = 0
    for idx,d in enumerate(data):
        diff += abs(predict(data[idx],model) - results[idx])
    return diff/len(results)


# input data is vector repr of ex(s)
# model is weight vector
def predict(input_data, model):
    return np.dot(input_data,model)

# init weight vec
def init_weights(feat_count):
    return np.random.random(feat_count)

# cost based on X data, Y target, model weights
def cost_func(X,Y,model):
    return (1/(2*len(Y))) * ((predict(X,model)-Y)**2)

def update_model(X,Y,model,learning_rate):
    preds = predict(X,model)
    error = Y-preds

    grads = np.dot(-X.T,error)

    grads /= len(X)

    grads *= learning_rate

    model -= grads
    return model


def add_bias_to_feats(data):
    bias = np.ones(shape=(len(data),1))
    return np.append(bias,data,axis=1)




def gradient_descent(X,Y, model, threshold=0.001, learning_rate=0.01, max_iter = 5000):
    cost = cost_func(X,Y,model)

    i=0

    while sum(cost) > threshold and i<max_iter:
        if i%10 == 0:
            print('iteration: '+str(i)+' ; cost: '+str(sum(cost)))
        model = update_model(X,Y,model, learning_rate)
        cost = cost_func(X,Y,model)
        i+=1

    return model


def main():
    datafile = 'slice_localization_data.csv'
    data = pd.read_csv(datafile)
    # print(data.values)
    # print(data.keys())
    # print(data[data.keys()[-1]])
    # print(data.values[:,-1])



    X = data.values[:,1:-2]
    X = normalize(X)

    cutoff = int(np.floor(3*(len(X))/4))
    Y = data.values[:,-1]
    Y = normalize_Y(Y)
    trainX = X[:cutoff]
    trainY = Y[:cutoff]

    valX = X[cutoff+1:]
    valY = Y[cutoff+1:]


    model = init_weights(len(X[0]))
    iter = 5000
    lr = 0.1

    model = gradient_descent(trainX,trainY,model,max_iter=iter,learning_rate=lr)
    store_model(model)

    test_preds = predict(valX,model)
    tuplist = []
    for idx,pred in enumerate(test_preds):

        print('prediction: '+str(pred)+' ; actual: '+str(valY[idx]))
        tuplist.append((pred,valY[idx]))
    with open('testres.csv','w') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        writer.writerow(tuplist)


    print(avg_diff(valX,valY,model))






if __name__ == "__main__":

    print(sys.argv)


    main()

