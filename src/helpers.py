import os
import main

import numpy as np

NUM_INPUT_LAYER  = 784
NUM_HIDDEN_LAYER = 100
NUM_OUTPUT_LAYER = 10
NUM_LAYERS = 3
NUM_EPOCH = 30
NUM_BATCH_SIZE = 10
NUM_LEARN_RATE = 2.5
TESTING = True

def vectorizeData(data):
    return zip((
        [np.reshape(matrix, (NUM_INPUT_LAYER, 1)) for matrix in data[0]],
        data[1]
    ))

def makeOutputLayer(answer):
    outputLayerUnitVector = np.zeros((10, 1))
    outputLayerUnitVector[answer] = 1.0
    return outputLayerUnitVector

def sigmoidActivationFunction(t):

    return 1.0/(1.0+np.exp(-t))

def sigmoidDerivative(t):
    sig = sigmoidActivationFunction(t)
    return sig*(1-sig)

def size(data):
    return data.get_value(borrow=True).shape[0] if main.USE_GPU else len(data)