"""
ANN Digit Recognition Program

Ziqiao Charlie Li
100832579
for AI (COMP4106) Final Project Winter 2016
Dr. Oommen
"""

import helpers as h
import main
import numpy as np

class RecognitionNetwork(object):
    def __init__(self, data):
        # Data idx: 1) training data, 2) validation data, 3) test data
        # Create layers with static settings above
        if len(data) != 3:
            raise ValueError('Invalid data loaded')

        self.loadData(data)
        self.generateRandomWeights()

    def train(self, epoch=30, batchSize=10, learnRate=3.0):
        if epoch < 1 or batchSize < 1: raise ValueError("Input error: epoch {0}, batch size {0}".format(epoch,batchSize))
        trainSample = self.myData[0]
        #validationSample, validationAnswer = self.myData[1]
        # Start generational training
        for i in xrange(epoch):
            np.random.shuffle(trainSample)
            batches = [trainSample[j:j+batchSize] for j in xrange(0, len(trainSample), batchSize)]
            for batch in batches:
                self.gradientDescent(batch)

    def gradientDescent(self, batch):
        pass

    def backpropagation(self):
        pass

    def costFunctionPrime(self):
        pass

    def test(self):
        testSample = self.myData[2], self.myData[2]

    def loadData(self, data):
        self.trainingData, self.validationData, self.testData = data
        if main.USE_GPU:
            self.myData = (h.loadIntoGPU(self.trainingData),
                           h.loadIntoGPU(self.validationData),
                           h.loadIntoGPU(self.myData))
        else:
            self.myData = (self.trainingData, self.validationData, self.testData)

    def generateRandomWeights(self):
         # Initialize Hidden Layer biases, followed by output layer biases (Gaussian distribution)
        self.layerBiases = [np.random.randn(h.NUM_HIDDEN_LAYER, 1), np.random.randn(h.NUM_OUTPUT_LAYER, 1)]
        self.layerOutputByHidden = np.random.randn(h.NUM_OUTPUT_LAYER, h.NUM_HIDDEN_LAYER)
        self.layerHiddenByInput = np.random.randn(h.NUM_HIDDEN_LAYER, h.NUM_INPUT_LAYER)
        # Initialize random weights for each layer. Idx 0) Input->Hidden layer mapping 1)
        self.layerWeights = [self.layerHiddenByInput,self.layerOutputByHidden]
        print "Generated weights"



