import os
import main
if main.USE_GPU:
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=gpu,floatX=float32"
else:
    os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu,floatX=float32"

import numpy as np
import theano


NUM_INPUT_LAYER  = 784
NUM_HIDDEN_LAYER = 30
NUM_OUTPUT_LAYER = 10
NUM_LAYERS = 3


def vectorizeData(data):
    trainingData = zip((
        [np.reshape(matrix, (NUM_INPUT_LAYER, 1)) for matrix in data[0]],
        [makeOutputLayer(answer) for answer in data[1]]
    ))
    return trainingData


def makeOutputLayer(answer):
    outputLayerUnitVector = np.zeros((10, 1))
    outputLayerUnitVector[answer] = 1.0
    return outputLayerUnitVector

def loadIntoGPU(data):
    # Allow GPU to store one instance of our data, when we need it, we borrow it from the GPU
    testElements = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    validationElement = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
    return testElements, theano.tensor.cast(validationElement, 'int32')

