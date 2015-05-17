import arff

import knn_wnn

instances = []

for row in arff.load('ionosphere.arff'):
    instances.append(row);

def loocv_classes(instances, k):
    correctedPredicted = 0
    for index in range(instances.__len__()):
        # Too expensive to copy the whole list
        newInstances = list(instances)
        newInstances.pop(index)
        if instances[index].Class == knn_wnn.classPrediction(newInstances, instances[index], k):
            correctedPredicted += 1
    print "Accuracy with LOOCV validation: "
    print correctedPredicted/float(instances.__len__())


loocv_classes(instances, 5)
