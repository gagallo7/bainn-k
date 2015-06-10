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


lastCommand = ""

# Parser for user's command
def parse ( c, num, op ):
    optArgs = True
    parsed = re.match(r'([cn]) * ([wk]).* (\d+).* ([biv]).* ([\.\d]+)', op, re.M|re.I )

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
                
        msg += "  || Ln-norm of " + parsed.group(5) + ", "
        details.append ( float ( parsed.group(5) ) )

        if ( parsed.group(4) == 'v' ):
            msg += "VDM enabled"
            details.append (2)
        elif ( parsed.group(4) == 'b' ):
            msg += "Binary distance for nominal"
            details.append (1)
        else:
            msg += "Ignore Nominal Attributes"
            details.append (0)

    command.append ( details )
    print msg

    if ( parsed.group(1)[0] == 'c' ):
        loocv_classes ( op, c, *command )
    else:
        loocv_classes ( op, num, *command )

def printHelp ():
    print "\nSyntax: <problem> <nn-type> <k-value> <nominal-dist> <Ln-norm>"
    print "Required parameters"
    print "\tproblem: 'c' for Classification, 'n' for Numerical Regression"
    print "\tnn-type: 'w' for w-nn, 'k' for k-nn"
    print "\tk-value: integer >= 1"
    print "Optional parameters (Both required)"
    print "\tnominal-dist: 'b' for binary distance 'v' for VDM or 'i' to ignore nominal attributes. (default 'i')"
    print "\t\tUse binary or value difference measure to calculate distance among nominal attributes"
    print "\tLn-norm: real number >= 0. (default 2.0)" 
    print "\t\tUsed to calculate distances."
    print "\t"
        
# Make the program flexible to user's input
def iterativeMode () :
    op = ''

    printHelp()

    while ( op != 'q' ):
        # The given problems
        c = Classification("ionosphere.arff")
        num = NumericalPrediction("autos3.arff")

        op = raw_input("Command ('q' to quit, 'h' for help): ")

        if ( op == 'q' ):
            break
        elif ( op == 'h' ):
            printHelp()
            continue
        elif op:
            parse (c, num, op)
        
    print ''

# Leave-one-out-cross-validation
def loocv_classes(command, problem, k, type='knn', details = []):

    # Too expensive to copy the whole list
    newInstances = deque(problem.instances)

    for index in range(problem.instances.__len__()):

        # Getting the new training set without one instance
        testInstance = newInstances.popleft() # first instance coming out of list

        problem.evaluate ( testInstance, newInstances, k, type, details )
        newInstances.append ( testInstance ) # same instance comes back as the last element

    problem.endEvaluation()
    problem.printError()
    problem.logError( command )



# Activating iterative mode
iterativeMode()
