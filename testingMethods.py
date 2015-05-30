# Falcon team
# Guilherme Alcarde Gallo       z5030891
# Pedro Lucas Albuquerque       z5046915

# Based on
# http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# Using the external module arff 0.9
# URL: https://pypi.python.org/pypi/arff/0.9

import arff

from knn_wnn import *

from collections import deque

# Leave-one-out-cross-validation
def loocv_classes(problem, k):

    # Too expensive to copy the whole list
    newInstances = deque(problem.instances)

    for index in range(problem.instances.__len__()):

        # Getting the new training set without one instance
        testInstance = newInstances.popleft() # first instance coming out of list

        problem.evaluate ( testInstance, newInstances, k )
        newInstances.append ( testInstance ) # same instance comes back as the last element

    problem.endEvaluation()
    problem.printError()


# The given problems
c = Classification("ionosphere.arff")
num = NumericalPrediction("autos2.arff")

print "Testing classification data..."
loocv_classes( c, 5 )

print "\nTesting numerical data..."
loocv_classes( num, 5 )
