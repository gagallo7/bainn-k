# Falcon team
# Guilherme Alcarde Gallo       z5030891
# Pedro Lucas Albuquerque       z5046915

# Based on
# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

import arff
import math
import operator
import heapq

def defineClass_kNN ( neighbours ):
    # tuples class and counter
        classes = []

    for x in neighbours:
        classes [ x[1] ] = classes [ x[1] ] + 1

    return classes


def getNeighbours ( trainingSet, testInstance, k ):
        dist = []
        neighbours = []
        length = len ( testInstance ) -1

        for x in range ( len ( trainingSet ) ):
            dist.append ( ( eucDist ( trainingSet [x], testInstance, length ), testInstance.Class ) )

        heapq.heapify ( dist )

        return heapq.nsmallest ( k, dist )

def eucDist ( i1, i2, length ):
    dist = 0

    for x in range ( length ):
        dist += pow ( ( i1[x] - i2[x] ), 2 )
    return math.sqrt ( dist )

instances = []

for row in arff.load ('ionosphere.arff' ):
   instances.append ( row ); 

print instances[1].Class
print getNeighbours ( instances, instances[0], 5 )
