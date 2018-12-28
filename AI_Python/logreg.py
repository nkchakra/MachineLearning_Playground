import sys
import os
import argparse
import numpy as np
import scipy.special as sp

sys.path.append(os.path.abspath('../utils'))
sys.path.append(os.path.abspath('../P3'))


from data_utils import *
from crossval import *
# from crossval_test import *
import time


from mldata import *

list_vec = []

# makes sure to throw an error on numerical issues with numpy
oldsettings = np.seterr(all='raise')
################################################
####                                        ####
####      Logistic Regression Functions     ####
####                                        ####
################################################


'''
    This function will operate gradient descent until the cost is low
'''


def gradient_descent(weights, data, learning_rate,threshold, iter_count):

    #print('\n')
    #print('learning rate: '+str(learning_rate))
    #print('cost threshold: '+str(threshold))
    cost = calc_cost(data, weights)
    N = len(data)
    #print('N: '+str(N))

    labels = data[:, -1]

    i = 0
    while (cost > threshold and i < iter_count):
        # train here
        i=i+1
        #print('grad desc iteration: '+str(i))
        predictions = predict(data, weights)
        grads = np.dot(data.T, (predictions - labels).T)
        grads = grads / N
        grads = grads * learning_rate
        weights = (weights - grads.T)[0, :]
        cost = calc_cost(data, weights)
        #print('cost: '+str(cost))

    return weights


'''
    This function will calculate the cost of our current concept against the input data, as well as calculate the weight gradients
    @param data: data to calculate cost over
    @param weights: weights we are getting gradients for
    NOTE: Now works with np.array
'''


def calc_cost(data, weights):
    m = len(data)
    totalCost = 0
    sum_err = 0
    epsilon = 1e-5
    predictions = predict(data, weights)
    # print('predics: '+str(predictions))
    labels = data[:, -1]
    c1 = -labels * np.log(predictions+epsilon)

    c2 = (1 - labels) * np.log(1 - predictions+epsilon)
    c = c1 - c2
    totalCost = c.sum()

    return (totalCost / m)


'''
    This function is the logistic function (AKA sigmoid function)
'''


def sigmoid(x):


    if x > 1000 or x < -1000:

        return sp.expit(x/100)
    else:
        return sp.expit(x)



'''
    This function predicts a label for each example in the data matrix with the current weights
    NOTE: Now works with np.array
'''


def predict(data, weights):
    predictions = np.zeros((1, len(data)))

    for i in range(0, len(data)):

        v = np.dot(data[i,:],weights)
        predictions[0, i] = sigmoid(v)

    return predictions


def label_predict(data,weights,isList=True,schema=None):
    #print('data'+str(data[0,1:len(data)]))

    if isList == False:

        return np.ceil(sigmoid(np.dot(data[1:len(data)],weights))-0.5)*2-1
        #
    else:
        return [np.ceil(p-0.5)*2 -1 for p in predict(data[:,1:len(data)],weights)][0]


#################################
####                         ####
####      Main Functions     ####
####                         ####
#################################

'''This function is the main entry point for Naive Bayes and handles all the argument and option setting'''



def logreg_main(args):
    schema, examples = parse_c45(args.datapath, rootdir="../440data")
    # print("schema: " + str(schema))
    print('\n\n')
    learning_rate = 1/len(examples)
    itercount = len(examples)
    # print('examples: \n'+str(examples[:2]))
    examples = convert_to_nparray(examples, schema)
    run_crossval = args.crossval
    if run_crossval == 0:
        if args.datapath == "example":
            k = 3
        else:
            k = 5
        #print('crossval, %s k' % k)
        cross_validation(examples,schema, generate_model, classify_model, k, False, False, 0, learning_rate,itercount)
    else:

        if args.datapath == 'example':
            learning_rate = 0.1
            itercount = 100
        if args.datapath == 'spam':
            learning_rate = 0.000001
            itercount = 8000
        print('whole set')
        print('learning rate: '+str(learning_rate))
        print('grad desc iter count: '+str(itercount))
        weights = generate_model(examples,schema,learning_rate,itercount)
        vec = classify_model(weights,examples,schema)
        
        full_example_metrics(vec)


def generate_model(examples,schema, alpha, iter_count):
    # print('examples[0]: '+str(len(examples[0])))
    # print(examples)
    # print('examples[:,1:13]: '+str(examples[:,1:13]))
    data = examples[:,1:len(examples[0])]
    weight_len = len(data[0])
    weights = np.zeros(weight_len,dtype='float')
    # print(weights)
    # print(calc_cost(data,weights))
    # print(predict(data, weights))
    weights = (gradient_descent(weights, data, alpha,0.01,iter_count))
    return weights
'''
    @param example: A single example (in numpy array form) to be classified 
    @param schema: The mldata.Schema that has the list of features
    @return confusion_mat: the confusion matrix of all classified examples in the form [TP, FN, FP, TN]
'''
def classify_model(weights, data, schema):
    data = data[:,1:len(data[0])]
    predictions = predict(data,weights)[0]
    og_predictions = predictions
    for i,pred in enumerate(predictions):
        if(pred > 0.5):
            predictions[i] = 1
        else:
            predictions[i] = 0

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    count = 0
    cwp = []
    for i in range(0,len(predictions)):
        estimate = -1
        og = -1 # this will be the confidence in prediction
        if(predictions[i] == 1):
            og = og_predictions[i]
        else:
            og = 1-og_predictions[i]
        cwp.append((predictions[i], data[i][-1],og))
        count += data[i][-1]
        if(predictions[i] == data[i][-1]):
            if(predictions[i] == 1):
                TP += 1
            else:
                TN += 1
        else:
            if(predictions[i] == 1):
                FP += 1
            else:
                FN += 1

    if(TP+TN+FN+FP == len(predictions)):
        #print(count/len(data))

        #print('conf_vec: '+str([TP,FN,FP,TN]))
        list_vec.append([TP,FN,FP,TN])
        return {"confusion":[TP,FN,FP,TN], 'classes_with_probs':cwp}

def parse_arguments():
    def check_nonnegative(value):
        ivalue = float(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("%s is an invalid negative value" % value)
        return ivalue

    parser = argparse.ArgumentParser(description='Make a Naive Bayes Classifier')
    parser.add_argument('datapath', help='path to the data (without file extensions)')
    parser.add_argument('crossval', type=int, choices=[0, 1], help='0: cross-validation, 1: full sample')
    parser.add_argument('alpha', type=check_nonnegative, help='real number that sets the learning rate')

    args = parser.parse_args()

    print(args)
    return args



if __name__ == "__main__":
    args = parse_arguments()    
    logreg_main(args)
