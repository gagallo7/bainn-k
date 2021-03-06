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

# Return the average of the k-nn values
def numericalPredictionKNN(neighbours):
    predictedNumber = 0

    for n in neighbours:
        predictedNumber += float ( n[1] )

    return predictedNumber/ float ( neighbours.__len__() )

# w-nn
def weightedClassification(neighbours):
    rank={}
    for neighbour in neighbours:
        # No Distance case, assume the same class
        if neighbour[0] == 0: return neighbour[1]
        if neighbour[1] not in rank:
            rank[neighbour[1]] = 1.0/(float(neighbour[0])**2)
        else:
            rank[neighbour[1]] += 1.0/(float(neighbour[0])**2)

    return max(rank, key=lambda key: rank[key])

# w-nn
# Return the weighted average of the k-nn values
def weightedNumericalPrediction(neighbours):
    totalWeight = 0
    sumElementTimesWeight = 0
    for neighbour in neighbours:
        if neighbour[0] == 0: return neighbour[1]
        weight = 1.0/(float(neighbour[0])**2)
        totalWeight += weight
        sumElementTimesWeight += neighbour[1] * weight

    return sumElementTimesWeight/totalWeight


# Defines who will be the neighbours for an instance
def getNeighbours(trainingSet, instance, k, details):
    dist = []
    length = len(instance) - 1
    dictProbs = {}

    if ( details != [] and details[1] == 2 ):
        dictProbs = probsForVDM (trainingSet)

    for x in range(len(trainingSet)):
        if details == []:
            dist.append((LnDist(trainingSet[x], instance, length, dictProbs), trainingSet[x]._values[-1]))
        else:
            dist.append((LnDist(trainingSet[x], instance, length, dictProbs, *details), trainingSet[x]._values[-1]))
                
    heapq.heapify(dist)

    return heapq.nsmallest(k, dist)

# L-n distance
def LnDist(i1, i2, length, dictProbs, n = 2, nominalEvaluation = 0 ):
    dist = 0
    n = float (n)
    # Treating Manhattan-Distance Ln = 0
    div = float (1/n) if n > 0 else 1

    for x in range(length):
        if type ( i1[x] ) != type ( str() ):
            dist += abs ( (i1[x] - i2[x])**n )
        elif nominalEvaluation > 0:
            if ( i1[x] != i2[x] ):
                if ( nominalEvaluation == 2 ):
                    dist += vdm(i1[x], i2[x], dictProbs, n)
                else:
                    dist += 1

    return float ( dist**div )

# Storing the probabilities attribute given class
# P(attr|class) used for VDM
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

# Value difference measure implementation
def vdm(val1, val2, dictProbs, n):
    total = 0
    for index, attrubutesPerClass in dictProbs.items():
        if(val1 in attrubutesPerClass) and (val2 in attrubutesPerClass):
            cost = abs(attrubutesPerClass[val1] - attrubutesPerClass[val2])**n
            #print "P(%s | %s) - P(%s | %s) = %f " %(index, val1, index, val2, cost)
            total += abs(attrubutesPerClass[val1] - attrubutesPerClass[val2])**n
    #print "Total: %f" % (total)
    return total


# Main classes

# Problem is a class that receives data from a Machine Learning
# problem
class Problem:
    def __init__(self, filename):
        self.instances = []
        self.errors = []
        self.k = 0

        for row in arff.load(filename):
            self.instances.append(row);

    # Normalizes the numerical attributes
    def normalizeNumericalAttr ( self ):
        # storing the minimum and the maximum value
        # of each attribute in dictionary
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

    # Given a test instance, 
    # it will return the predicted class/number
    def prediction( self, trainingSet, instance, k, details ):
        return
    # Used for cross-validation
    # Evaluates one test instance against the splitted training data
    def evaluate ( self, lhs, newInstances, k, details ):
        return
    # Do the final operations to get the correct evaluation
    def endEvaluation ( self ):
        return
    def printError ( self ):
        return

    # Store the results in the results.txt file
    def logError ( self, command = "" ):
        f = open ( 'results.txt', 'a' )
        if command != "":
            f.write ( command + ", " )
        for e in self.errors: f.write ( str(e) + ", " )
        f.write ( '\n' )
        f.close()

# Specializes Problem into a Classification problem
class Classification ( Problem ):

    def __init__(self, filename):
        self.accuracy = 0
        Problem.__init__ ( self, filename )     # super
        
    def prediction( self, trainingSet, instance, k, type, details ):
        if ( type == 'knn' ):
            predictedClass = defineClass_kNN(getNeighbours(trainingSet, instance, k, details))
        else:
            predictedClass = weightedClassification(getNeighbours(trainingSet, instance, k, details))
        return predictedClass;

    def evaluate ( self, lhs, newInstances, k, type='knn', details = [] ):
        rhs = self.prediction ( newInstances, lhs, k, type, details )
        self.accuracy += lhs._values[-1] == rhs

    def endEvaluation ( self ):
        self.accuracy = self.accuracy / float ( len ( self.instances ) )
        self.errors.append ( "{0:2.4f}".format ( self.accuracy*100 ) )

    def printError ( self ):
        accStr = "Accuracy: {}%".format ( self.errors[0] )
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
    def prediction( self, trainingSet, instance, k, type, details ):
        if ( type == 'knn' ):
            predictedNumber = numericalPredictionKNN(getNeighbours(trainingSet, instance, k, details))
        else:
            predictedNumber = weightedNumericalPrediction(getNeighbours(trainingSet, instance, k, details))
        return predictedNumber;

    def evaluate ( self, lhs, newInstances, k, type='knn', details = [] ):
        predicted = self.prediction ( newInstances, lhs, k, type, details )
        actual = float ( lhs._values[-1] )

        self.meanAbsError += abs ( actual - predicted )
        self.relativeAbsError += abs ( actual - self.average )
        self.rootedSquaredError += ( predicted - actual )**2
  
    def endEvaluation ( self ):
        # The relative and mean abs error have the numerator in common
        self.relativeAbsError = self.meanAbsError / self.relativeAbsError
        self.meanAbsError = self.meanAbsError / float ( len ( self.instances ) )
        self.rootedSquaredError = math.sqrt ( self.rootedSquaredError / float ( len ( self.instances ) ) )
        self.errors.append ( "" )   # there is no accuracy
        self.errors.append ( "{0:2.4f}".format ( self.meanAbsError ) )
        self.errors.append ( "{0:2.4f}".format ( self.relativeAbsError*100 ) )
        self.errors.append ( "{0:2.4f}".format ( self.rootedSquaredError ) )

    def printError ( self ):
        meanAbsoluteErrorStr =  "Mean absolute error:     {}".format ( self.errors[1] )
        relativeAbsErrorStr =   "Relative absolute error:   {}%".format ( self.errors[2] ) 
        rootedSquaredErrorStr = "Root Squared error:      {}".format ( self.errors[3] )
        print meanAbsoluteErrorStr
        print relativeAbsErrorStr
        print rootedSquaredErrorStr
# --------------------------------
