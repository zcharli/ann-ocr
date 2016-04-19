"""
ANN Digit Recognition Program

Ziqiao Charlie Li
100832579
for AI (COMP4106) Final Project Winter 2016
Dr. Oommen
"""
from __future__ import division
USE_GPU = False
import numpy as np
from mnist import MNIST
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import RecognitionNetwork as net
import helpers as h

n = 0
def loadData():
    # mndata = MNIST('../data')
    # train = mndata.load_training()
    # test = mndata.load_testing()
    # print len(train)
    # print len(test)
    # data = normalizeDataset(train, test)
    # d = {"testing": data[1], "training": data[0]}
    # saved = gzip.open(r'..\data\mnist.pklz', 'wb')
    # pickle.dump(d, saved)
    # saved.close()
    with gzip.open("../data/mnist.pklz") as f:
        data = prepareMNISTData(pickle.load(f))
    # with open("../data/vanilla.pkl") as f:
    #     data = preparePickledData(pickle.load(f))
    annRecognitionNetwork = net.RecognitionNetwork(data, False)
    annRecognitionNetwork.train(h.NUM_EPOCH, h.NUM_BATCH_SIZE, h.NUM_LEARN_RATE)
    global n
    n = annRecognitionNetwork

def normalizeDataset(train, test):
    train = (np.asarray(map(lambda x: map(lambda y: float(y/255), x), train[0])), np.asarray(train[1]))
    test = (np.asarray(map(lambda x: map(lambda y: float(y/255), x), test[0])), np.asarray(test[1]))
    return train, test

def preparePickledData(data):
    return data["layerHiddenBiases"], data["layerOutputBiases"], \
           data["layerOutputToHiddenWeights"], data["layerHiddenToInputWeights"]

def prepareMNISTData(data):
    reshape = np.reshape
    inputs = map(lambda x: reshape(x, (784, 1)), data["training"][0])
    vectorize = vectorized_result
    answers = map(lambda x: vectorize(x), data["training"][1])
    train = zip(inputs, answers)
    testInputs = map(lambda x: reshape(x, (784, 1)), data["testing"][0])
    test = zip(testInputs, data["testing"][1])
    print len(train)
    return train, test

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == '__main__':
    loadData()