import arff

import knn_wnn

from collections import deque

instances = []

for row in arff.load('ionosphere.arff'):
    instances.append(row);

# Leave-one-out-cross-validation
def loocv_classes(instances, k):
    correctedPredicted = 0

    # Too expensive to copy the whole list
    newInstances = deque(instances)

    for index in range(instances.__len__()):

        # Getting the new training set without
        # one instance 
        testInstance = newInstances.popleft() # first instance coming out of list

        if testInstance.Class == knn_wnn.classPrediction(newInstances, testInstance, k):
            correctedPredicted += 1

        newInstances.append ( testInstance ) # same instance comes back as the last element

    print "Accuracy with LOOCV validation: "
    print correctedPredicted/float(instances.__len__())


loocv_classes(instances, 5)
