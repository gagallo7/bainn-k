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

import re

def parse ( c, num, op ):
    parsed = re.match(r'([cn]) * ([wk]).* (\d*)', op, re.M|re.I )

    if parsed == None:
        raise NameError('Bad command syntax')

    command = []
    msg = ""

    command.append ( int ( parsed.group(3) ) )

    if ( parsed.group(2)[0] == 'w' ):
        msg += "Using WNN "
        command.append ( 'wnn' )
    else:
        msg += "Using KNN "
        command.append ( 'knn' )

    msg += "with k = {}".format ( int ( parsed.group (3) ) )

    print msg

    if ( parsed.group(1)[0] == 'c' ):
        loocv_classes ( c, *command )
    else:
        loocv_classes ( num, *command )

# Leave-one-out-cross-validation
def loocv_classes(problem, k, type='knn'):
    print "Nearest neighbour type: {}".format ( type )

    # Too expensive to copy the whole list
    newInstances = deque(problem.instances)

    for index in range(problem.instances.__len__()):

        # Getting the new training set without one instance
        testInstance = newInstances.popleft() # first instance coming out of list

        problem.evaluate ( testInstance, newInstances, k, type )
        newInstances.append ( testInstance ) # same instance comes back as the last element

    problem.endEvaluation()
    problem.printError()


# The given problems
c = Classification("ionosphere.arff")
num = NumericalPrediction("autos3.arff")

#test = [5,'wnn']

#loocv_classes ( c, *test )

op = ''

while ( op != 'q' ):
    print "\nSyntax: <problem> <nn-type> <k-value>"
    print "\tproblem: c for Classification, n for Numerical Regression"
    print "\t\tnn-type: w for w-nn, k for k-nn"
    print "\t\t\tk-value: integer >= 1"

    op = raw_input("Command: ")

    parse (c, num, op)
    
        
'''
print "Testing classification data..."
loocv_classes( c, 5, 'wnn' )

print "\nTesting numerical data..."
loocv_classes( num, 5, 'wnn' )

print "\nNormalizing..."
c.normalizeNumericalAttr()
num.normalizeNumericalAttr()

print "Testing classification data..."
loocv_classes( c, 5, 'wnn' )

print "\nTesting numerical data..."
loocv_classes( num, 5, 'wnn' )
'''
