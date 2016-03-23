"""
ANN Digit Recognition Program

Ziqiao Charlie Li
100832579
for AI (COMP4106) Final Project Winter 2016
Dr. Oommen
"""

USE_GPU = False

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import RecognitionNetwork as net
import helpers as h

n = 0
def loadData():
    with gzip.open("../data/mnist.pkl.gz") as f:
        data = prepareMNISTData(pickle.load(f))
    annRecognitionNetwork = net.RecognitionNetwork(data)
    annRecognitionNetwork.train(h.NUM_EPOCH, h.NUM_BATCH_SIZE, h.NUM_LEARN_RATE)
    global n
    n = annRecognitionNetwork

def prepareMNISTData(data):
    tr_d, va_d, te_d = data
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    print len(training_data)
    return training_data, validation_data, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    loadData()