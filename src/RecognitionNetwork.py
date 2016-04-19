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
import cPickle as pickle

class RecognitionNetwork(object):
    def __init__(self, data, pickled=False):
        # Data idx: 1) training data, 2) validation data, 3) test data
        # Create layers with static settings above
        if len(data) < 2:
            raise ValueError('Invalid data loaded')
        self.loadData(data, pickled)

    def loadData(self, data, pickled):
        if not pickled:
            self.trainingData, self.testData = data
            self.myData = (self.trainingData, self.testData)
            self.generateRandomWeights()
        else:
            self.layerHiddenBiases = data[0]
            self.layerOutputBiases = data[1]
            self.layerHiddenToOutput = data[2]
            self.layerInputToHidden = data[3]

    def train(self, epoch=30, batchSize=10, learnRate=3.0):
        if epoch < 1 or batchSize < 1: raise ValueError(
            "Input error: epoch {0}, batch size {0}".format(epoch, batchSize))
        self.learnRate = learnRate
        # validationSample, validationAnswer = self.myData[1]
        # Start generational training
        length = len(self.trainingData)
        testLength = len(self.testData)
        for i in xrange(epoch):
            np.random.shuffle(self.trainingData)
            batches = [self.trainingData[j:j + batchSize] for j in xrange(0, length, batchSize)]
            for batch in batches:
                self.gradientDescent(batch)
            if h.TESTING:
                print "Epoch {0}: {1} / {2}".format(
                    i, self.test(self.testData), testLength)
            else:
                print "Epoch {0} complete".format(j)
        # d = {"layerHiddenBiases": self.layerHiddenBiases, "layerOutputBiases": self.layerOutputBiases,
        #      "layerOutputToHiddenWeights": self.layerOutputToHiddenWeights,
        #      "layerHiddenToInputWeights" : self.layerHiddenToInputWeights}
        # saved = open(r'..\data\vanilla.pkl', 'wb')
        # pickle.dump(d, saved)
        # saved.close()

    def gradientDescent(self, batch):
        # Generate a container matrix to aggregate error
        aggregateHiddenLayerBiases = np.zeros(self.layerHiddenBiases.shape)
        aggregateOutputLayerBiases = np.zeros(self.layerOutputBiases.shape)
        aggregateHiddenLayerWeight = np.zeros(self.layerInputToHidden.shape)
        aggregateOutputLayerWeight = np.zeros(self.layerHiddenToOutput.shape)
        for sample, answer in batch:
            bpHiddenBias, bpOutputBias, bpHiddenWeight, bpOutputWeight = self.backpropagate(sample, answer)
            aggregateHiddenLayerBiases += bpHiddenBias
            aggregateHiddenLayerWeight += bpHiddenWeight
            aggregateOutputLayerBiases += bpOutputBias
            aggregateOutputLayerWeight += bpOutputWeight
        self.layerOutputBiases -= (self.learnRate / len(batch)) * aggregateOutputLayerBiases
        self.layerHiddenBiases -= (self.learnRate / len(batch)) * aggregateHiddenLayerBiases
        self.layerInputToHidden -= (self.learnRate / len(batch)) * aggregateHiddenLayerWeight
        self.layerHiddenToOutput -= (self.learnRate / len(batch)) * aggregateOutputLayerWeight

    def backpropagate(self, sample, answer):
        # Store all the possible activations for this pass through
        weightedHiddenLayer = np.dot(self.layerInputToHidden, sample) + self.layerHiddenBiases
        activationHiddenLayer = h.sigmoidActivationFunction(weightedHiddenLayer)
        weightedOutputLayer = np.dot(self.layerHiddenToOutput, activationHiddenLayer) + self.layerOutputBiases
        activationOutputLayer = h.sigmoidActivationFunction(weightedOutputLayer)
        # Compute the error of the quad cost based on hidden & output layer rate of change * (...)
        #     Hadamard Product :(activation^l - answer)*(how fast activation func is changing)
        # For reference our quadratic cost is 1/2 * \sum_{j}(answer_j - activation_j)^2 we get its derivative
        # Begin to back propagate the error from our output layer
        outputLayerBiasError = (activationOutputLayer - answer) * h.sigmoidDerivative(weightedOutputLayer)
        # Layer 2 weighted input transposed
        outputLayerWeightError = np.dot(outputLayerBiasError, np.array(activationHiddenLayer).T)
        hiddenLayerBiasError = np.dot(np.array(self.layerHiddenToOutput).T,
                                      outputLayerBiasError) * h.sigmoidDerivative(weightedHiddenLayer)
        hiddenLayerWeightError = np.dot(hiddenLayerBiasError, np.array(sample).T)
        # Now that we propagated back to the input layer, lets update our weights
        return hiddenLayerBiasError, outputLayerBiasError, hiddenLayerWeightError, outputLayerWeightError

    def generateRandomWeights(self):
        # Initialize Hidden Layer biases, followed by output layer biases (Gaussian distribution)
        # self.layerBiases = [np.random.randn(h.NUM_HIDDEN_LAYER, 1), np.random.randn(h.NUM_OUTPUT_LAYER, 1)]
        self.layerHiddenBiases = np.random.randn(h.NUM_HIDDEN_LAYER, 1)
        self.layerOutputBiases = np.random.randn(h.NUM_OUTPUT_LAYER, 1)
        self.layerHiddenToOutput = np.random.randn(h.NUM_OUTPUT_LAYER, h.NUM_HIDDEN_LAYER)
        self.layerInputToHidden = np.random.randn(h.NUM_HIDDEN_LAYER, h.NUM_INPUT_LAYER)
        # Initialize random weights for each layer. Idx 0) Input->Hidden layer mapping 1)
        # self.layerWeights = [self.layerHiddenToInputWeights, self.layerOutputToHiddenWeights]
        print "Generated weights"

    def test(self, testData):
        # For each test data ([image], answer), pass it thru our ANN
        # Take the largest output value index and check if they equal the answer
        testResults = map(lambda x: (np.argmax(self.feedforward(x[0])), x[1]), testData)
        return sum(int(x == y) for (x, y) in testResults)

    def feedforward(self, sampleActivation):
        # Pass the sample through from input to our hidden layer
        sampleActivation = h.sigmoidActivationFunction(np.dot(self.layerInputToHidden, sampleActivation) + self.layerHiddenBiases)
        # Lastly pass it out from our output layer
        sampleActivation = h.sigmoidActivationFunction(np.dot(self.layerHiddenToOutput, sampleActivation) + self.layerOutputBiases)
        return sampleActivation