"""
ANN Digit Recognition Program

Ziqiao Charlie Li
100832579
for AI (COMP4106) Final Project Winter 2016
Dr. Oommen
"""
import numpy
try:
    import cPickle as pickle
except:
    import pickle
import gzip


def loadData():
    with gzip.open("../data/mnist.pkl.gz") as f:
        training_data, validation_data, test_data = pickle.load(f)


if __name__ == '__main__':
    loadData()