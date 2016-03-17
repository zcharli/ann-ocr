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

        self.trainingData = h.vectorizeData(data[0])
        self.validationData = h.vectorizeData(data[1])
        self.testData = h.vectorizeData(data[2])
        if main.USE_GPU:
            self.myData = [h.loadIntoGPU(self.trainingData),
                           h.loadIntoGPU(self.validationData),
                           h.loadIntoGPU(self.testData)]
        else:
            self.myData = [self.trainingData, self.validationData, self.testData]
        # Initialize Hidden Layer biases, followed by output layer biases (Gaussian distribution)
        self.layerBiases = [np.random.randn(h.NUM_HIDDEN_LAYER, 1), np.random.randn(h.NUM_OUTPUT_LAYER, 1)]
        self.layerOutputByHidden = np.random.randn(h.NUM_OUTPUT_LAYER, h.NUM_HIDDEN_LAYER)
        self.layerHiddenByInput = np.random.randn(h.NUM_HIDDEN_LAYER, h.NUM_INPUT_LAYER)
        # Initialize random weights for each layer. Idx 0) Input->Hidden layer mapping 1)
        self.layerWeights = [self.layerHiddenByInput,self.layerOutputByHidden]

    def train(self):
        pass


