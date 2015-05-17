# Falcon team
# Guilherme Alcarde Gallo       z5030891
# Pedro Lucas Albuquerque       z5046915

# Based on
# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/


import math
import heapq
from collections import Counter


def defineClass_kNN(neighbours):
    return Counter(elem[1] for elem in neighbours).most_common()[0][0];


def getNeighbours(trainingSet, instance, k):
    dist = []
    length = len(instance) - 1

    for x in range(len(trainingSet)):
        dist.append((eucDist(trainingSet[x], instance, length), trainingSet[x].Class))

    heapq.heapify(dist)

    return heapq.nsmallest(k, dist)


def eucDist(i1, i2, length):
    dist = 0

    for x in range(length):
        dist += pow((i1[x] - i2[x]), 2)
    return math.sqrt(dist)


# Main functions

def numericalPrediction(trainingSet, instance, k):
    return predictedNumber;


def classPrediction(trainingSet, instance, k):
    predictedClass = defineClass_kNN(getNeighbours(trainingSet, instance, k))
    return predictedClass;


# --------------------------------
