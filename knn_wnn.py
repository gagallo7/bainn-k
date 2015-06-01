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

# Return the most common class from the k-nearest neighbours
def defineClass_kNN(neighbours):
    return Counter(elem[1] for elem in neighbours).most_common()[0][0];

# Defines who will be the neighbours for an instance
def getNeighbours(trainingSet, instance, k):
    dist = []
    length = len(instance) - 1

    for x in range(len(trainingSet)):
        dist.append((eucDist(trainingSet[x], instance, length), trainingSet[x]._values[-1]))

    heapq.heapify(dist)

    return heapq.nsmallest(k, dist)

# Euclidean distance
def eucDist(i1, i2, length):
    dist = 0

    for x in range(length):
        if type ( i1[x] ) != type ( str() ):
            dist += abs ( (i1[x] - i2[x])**16.0 )
        else:
            if ( i1[x] != i2[x] ):
                dist += 1

    #return math.sqrt(dist)
    return float ( dist**(1/16.0) )


# Main classes

# Problem is a class that receives data from a Machine Learning
# problem
class Problem:
    def __init__(self, filename):
        self.instances = []

        for row in arff.load(filename):
            self.instances.append(row);

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
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.fp = 0
        Problem.__init__ ( self, filename )     # super
        
    def prediction( self, trainingSet, instance, k):
        predictedClass = defineClass_kNN(getNeighbours(trainingSet, instance, k))
        return predictedClass;

    def evaluate ( self, lhs, newInstances, k ):
        rhs = self.prediction ( newInstances, lhs, k )
        self.accuracy += lhs._values[-1] == rhs

    def endEvaluation ( self ):
        self.accuracy = self.accuracy / float ( len ( self.instances ) )

    def printError ( self ):
        accStr = "Accuracy: {}".format ( self.accuracy )
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
    # k nearest neighbours
    def prediction( self, trainingSet, instance, k):
        predictedNumber = 0

        for n in getNeighbours(trainingSet, instance, k):
            predictedNumber += float ( n[1] )
        
        predictedNumber = predictedNumber / float ( k )

        return predictedNumber;

    def evaluate ( self, lhs, newInstances, k ):
        rhs = self.prediction ( newInstances, lhs, k )

        self.meanAbsError += abs ( float ( lhs._values[-1] ) - rhs )
        self.relativeAbsError += abs ( rhs - self.average )
        self.rootedSquaredError += ( rhs - self.average )**2

    def endEvaluation ( self ):
        # The relative and mean abs error have the numerator in common
        self.relativeAbsError = self.meanAbsError / self.relativeAbsError
        self.meanAbsError = self.meanAbsError / float ( len ( self.instances ) )

        self.rootedSquaredError = math.sqrt ( self.rootedSquaredError / float ( len ( self.instances ) ) )

    def printError ( self ):
        meanAbsoluteErrorStr = "Mean absolute error: {}".format ( self.meanAbsError )
        relativeAbsErrorStr = "Relative absolute error: {}".format ( self.relativeAbsError )
        rootedSquaredErrorStr = "Rooted Squared error: {}".format ( self.rootedSquaredError )
        print meanAbsoluteErrorStr
        print relativeAbsErrorStr
        print rootedSquaredErrorStr
# --------------------------------
