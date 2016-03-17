"""
ANN Digit Recognition Program

Ziqiao Charlie Li
100832579
for AI (COMP4106) Final Project Winter 2016
Dr. Oommen
"""

USE_GPU = True

import numpy
try:
    import cPickle as pickle
except:
    import pickle
import gzip
import RecognitionNetwork as net


def loadData():
    with gzip.open("../data/mnist.pkl.gz") as f:
        data = pickle.load(f)
    annRecognitionNetwork = net.RecognitionNetwork(data)
    annRecognitionNetwork.train()

if __name__ == '__main__':
    loadData()