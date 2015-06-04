# Falcon team
# Guilherme Alcarde Gallo       z5030891
# Pedro Lucas Albuquerque       z5046915

# Based on
# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# Using the external module arff 0.9
# URL: https://pypi.python.org/pypi/arff/0.9

import arff
import math
import heapq
from collections import Counter
from copy import deepcopy

# Avoiding division by zero
EPSILON = 1e-7

# Return the most common class from the k-nearest neighbours
def defineClass_kNN(neighbours):
    return Counter(elem[1] for elem in neighbours).most_common()[0][0];

# w-nn
def weightedClassification(neighbours):
    rank={}
    for neighbour in neighbours:
        if neighbour[0] == 0: return neighbour[1]
        if neighbour[1] not in rank:
            rank[neighbour[1]] = 1.0/(float(neighbour[0])**2)
        else:
            rank[neighbour[1]] += 1.0/(float(neighbour[0])**2)

    return max(rank, key=lambda key: rank[key])

# w-nn
def weightedNumericalPrediction(neighbours):
    totalWeight = 0
    sumElementTimesWeight = 0
    for neighbour in neighbours:
        if neighbour[0] == 0: return neighbour[1]
        weight = 1.0/(float(neighbour[0])**2)
        totalWeight += weight
        sumElementTimesWeight += neighbour[1] * weight

    return sumElementTimesWeight/totalWeight

def numericalPrediction(neighbours):
    predictedNumber = 0

    for n in neighbours:
        predictedNumber += float ( n[1] )

    return predictedNumber/ float ( neighbours.__len__() )

# Defines who will be the neighbours for an instance
def getNeighbours(trainingSet, instance, k):
    dist = []
    length = len(instance) - 1

    dictProbs = probsForVDM (trainingSet)

    for x in range(len(trainingSet)):
        dist.append((eucDist(trainingSet[x], instance, length, dictProbs), trainingSet[x]._values[-1]))

    #print dist
    heapq.heapify(dist)

    #print heapq.nsmallest(k, dist)
    return heapq.nsmallest(k, dist)

# Euclidean distance
def eucDist(i1, i2, length, dictProbs):
    dist = 0

    for x in range(length):
        if type ( i1[x] ) != type ( str() ):
            dist += abs ( (i1[x] - i2[x])**2.0 )
            pointo = (abs ( (i1[x] - i2[x])**2.0 ))
            #print "Points: (%f %f) %s"% (i1[x], i2[x], pointo)
        else:

            if ( i1[x] != i2[x] ):
                dist += 1
                #vdmresult = vdm(i1[x], i2[x], dictProbs, 2)
                #dist += vdmresult
                #dist += vdm(i1[x], i2[x], dictProbs, 2)

    #return math.sqrt(dist)
    return float ( dist**(1/2.0) )

def probsForVDM (trainingSet):
    dictProbs = {}
    numberValues = {}

    for tuple in trainingSet:
        for k, attr in enumerate(tuple._values[:-1]):
            if type ( attr ) == type ( str() ):
                tClass = tuple._values[-1]
                if attr not in numberValues:
                    numberValues[attr] = 1
                else:
                    numberValues[attr] += 1

                # Number of value/class
                if tClass not in dictProbs:
                    dictProbs[tClass] = {}
                    dictProbs[tClass][attr] = 1
                elif attr not in dictProbs[tClass]:
                    dictProbs[tClass][attr] = 1
                else:
                    dictProbs[tClass][attr] += 1

    # Prob = number of values with class / total number of values
    for k,attrubutesPerClass in dictProbs.items():
        for key, attribute in attrubutesPerClass.items():
            attrubutesPerClass[key] /= numberValues[key] * 1.0

    # print dictProbs
    return dictProbs

def vdm(val1, val2, dictProbs, n):
    total = 0
    for index, attrubutesPerClass in dictProbs.items():
        if(val1 in attrubutesPerClass) and (val2 in attrubutesPerClass):
            cost = abs(attrubutesPerClass[val1] - attrubutesPerClass[val2])**n*1.0
            #print "P(%s | %s) - P(%s | %s) = %f " %(index, val1, index, val2, cost)
            total += abs(attrubutesPerClass[val1] - attrubutesPerClass[val2])**n*1.0
    #print "Total: %f" % (total)
    return total


# Main classes

# Problem is a class that receives data from a Machine Learning
# problem
class Problem:
    def __init__(self, filename):
        self.instances = []

        for row in arff.load(filename):
            self.instances.append(row);

    # Normalizes the numerical attributes
    def normalizeNumericalAttr ( self ):
        # storing the minimum and the maximum value
        # of each attribute in dictionary
        #minValues = self.instances[0]._values[:]
        #maxValues = self.instances[0]._values[:]
        minValues = deepcopy ( self.instances[0]._values )
        maxValues = deepcopy ( self.instances[0]._values )

        # finding the minimum and the maximum
        for tuple in self.instances:
            for k, attr in enumerate(tuple._values[:-1]):
                if type ( attr ) != type ( str() ):
                    minValues[k] = min ( minValues[k], attr )
                    maxValues[k] = max ( maxValues[k], attr )

        # applying normalization
        # x' = (x-min)/(max-min)
        for tuple in self.instances:
            for k, attr in enumerate(tuple._values[:-1]):
                if type ( attr ) != type ( str() ):
                    tuple._values[k] = (tuple._values[k]-minValues[k])/(EPSILON+maxValues[k]-minValues[k])

        return


    def prediction( self, trainingSet, instance, k):
        return
    # Used for cross-validation
    # Evaluates one test instance against the splitted training data
    def evaluate ( self, lhs, newInstances, k ):
        return
    # Do the final operations to get the correct evaluation
    def endEvaluation ( self ):
        return
    def printError ( self ):
        return

# Specializes Problem into a Classification problem
class Classification ( Problem ):

    def __init__(self, filename):
        self.accuracy = 0
        Problem.__init__ ( self, filename )     # super
        
    def prediction( self, trainingSet, instance, k, type):
        if ( type == 'knn' ):
            predictedClass = defineClass_kNN(getNeighbours(trainingSet, instance, k))
        else:
            predictedClass = weightedClassification(getNeighbours(trainingSet, instance, k))
        return predictedClass;

    def evaluate ( self, lhs, newInstances, k, type='knn' ):
        rhs = self.prediction ( newInstances, lhs, k, type )
        self.accuracy += lhs._values[-1] == rhs

    def endEvaluation ( self ):
        self.accuracy = self.accuracy / float ( len ( self.instances ) )

    def printError ( self ):
        accStr = "Accuracy: {0:2.4f}%".format ( self.accuracy*100 )
        print accStr

# Specializes Problem into a Regression problem
class NumericalPrediction ( Problem ):
    def __init__(self, filename):
        self.meanAbsError = 0
        self.relativeAbsError = 0
        self.rootedSquaredError = 0
        Problem.__init__ ( self, filename )
        
        # Used for relative absolute error
        self.average = sum ([ float(c._values[-1]) for c in self.instances ]) / len ( self.instances )

    # The predicted number is the average of the 
    # k nearest neighbours or wnn
    def prediction( self, trainingSet, instance, k, type ):
        if ( type == 'knn' ):
            predictedNumber = numericalPrediction(getNeighbours(trainingSet, instance, k))
        else:
            predictedNumber = weightedNumericalPrediction(getNeighbours(trainingSet, instance, k))
        return predictedNumber;

    def evaluate ( self, lhs, newInstances, k, type='knn' ):
        rhs = self.prediction ( newInstances, lhs, k, type )

        self.meanAbsError += abs ( float ( lhs._values[-1] ) - rhs )
        self.relativeAbsError += abs ( rhs - self.average )
        self.rootedSquaredError += ( rhs - self.average )**2

    def endEvaluation ( self ):
        # The relative and mean abs error have the numerator in common
        self.relativeAbsError = self.meanAbsError / self.relativeAbsError
        self.meanAbsError = self.meanAbsError / float ( len ( self.instances ) )

        self.rootedSquaredError = math.sqrt ( self.rootedSquaredError / float ( len ( self.instances ) ) )

    def printError ( self ):
        meanAbsoluteErrorStr =  "Mean absolute error:     {0:2.4f}".format ( self.meanAbsError )
        relativeAbsErrorStr =   "Relative absolute error:   {0:2.4f}%".format ( self.relativeAbsError*100 )
        rootedSquaredErrorStr = "Root Squared error:      {0:2.4f}".format ( self.rootedSquaredError )
        print meanAbsoluteErrorStr
        print relativeAbsErrorStr
        print rootedSquaredErrorStr
# --------------------------------
