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

# Parser for user's command
def parse ( c, num, op ):
    optArgs = True
    parsed = re.match(r'([cn]) * ([wk]).* (\d+).* ([yn]).* ([\.\d]+)', op, re.M|re.I )

    # Without optional args
    if parsed == None:
        parsed = re.match(r'([cn]) * ([wk]).* (\d+)', op, re.M|re.I )
        optArgs = False

    # Error
    if parsed == None:
        raise NameError('Bad command syntax')

    if int ( parsed.group(3) ) < 1:
        raise NameError('Bad k value')

    command = []
    msg = ""
    details = []

    command.append ( int ( parsed.group(3) ) )

    if ( parsed.group(2)[0] == 'w' ):
        msg += "Using W-NN "
        command.append ( 'wnn' )
    else:
        msg += "Using K-NN "
        command.append ( 'knn' )

    msg += "with k = {}".format ( int ( parsed.group (3) ) )

    if ( optArgs ):
        if float ( parsed.group(5) ) < 0:
            raise NameError('Bad Ln value: Ln < 0')
                
        msg += "|| Ln-norm of " + parsed.group(5) + ", "
        details.append ( float ( parsed.group(5) ) )

        if ( parsed.group(4) == 'y' ):
            msg += " VDM enabled"
            details.append (True)
        else:
            msg += " VDM disabled"
            details.append (False)

    command.append ( details )
    print msg

    if ( parsed.group(1)[0] == 'c' ):
        loocv_classes ( c, *command )
    else:
        loocv_classes ( num, *command )

# Make the program flexible to user's input
def iterativeMode () :
    op = ''

    while ( op != 'q' ):
        print "\nSyntax: <problem> <nn-type> <k-value> <VDM> <Ln-norm>"
        print "Required parameters"
        print "\tproblem: 'c' for Classification, 'n' for Numerical Regression"
        print "\tnn-type: 'w' for w-nn, 'k' for k-nn"
        print "\tk-value: integer >= 1"
        print "Optional parameters (Both required)"
        print "\tVDM: 'y' or 'n'. (default 'n')"
        print "\t\tUse value difference measure to calculate distance among nominal attributes"
        print "\tLn-norm: real number >= 0. (default 2.0)" 
        print "\t\tUsed to calculate distances."
        print "\t"

        op = raw_input("Command ('q' to quit): ")

        if ( op == 'q' ):
            break
        elif op:
            parse (c, num, op)
        
    print ''

# Leave-one-out-cross-validation
def loocv_classes(problem, k, type='knn', details = []):

    # Too expensive to copy the whole list
    newInstances = deque(problem.instances)

    for index in range(problem.instances.__len__()):

        # Getting the new training set without one instance
        testInstance = newInstances.popleft() # first instance coming out of list

        problem.evaluate ( testInstance, newInstances, k, type, details )
        newInstances.append ( testInstance ) # same instance comes back as the last element

    problem.endEvaluation()
    problem.printError()


# The given problems
c = Classification("ionosphere.arff")
num = NumericalPrediction("autos3.arff")

# Activating iterative mode
iterativeMode()
