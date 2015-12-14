'''
Name: Cooper Bates
UNI: cbb2153

Fufills the specifications of Nearest Neighbor assignment part 2
'''

import numpy as np
from scipy.spatial import distance
from statistics import mode


def partition_data(p, data, index):
    '''
    Partitions total data chunk based on p value as well as iteration
    index
    '''
    
    # sets partition value
    partition = data.shape[0]/p    
    
    # set start and end row indicies for partition
    start = round(partition * index)
    end =  round(partition * (index + 1))
        
    # obtain training and test data
    training = np.delete(data, range(start, end) , 0)
    test = data[range(start, end)]
    
    return training, test


def n_validator(data, p, classifier, *args):
    '''
    Takes in a total data set, and partitions it in several different ways,
    testing the classifier on the data during each partition
    '''

    total = 0
    for i in range(p):
        training = partition_data(p, data, i)[0]
        test = partition_data(p, data, i)[1]

        class_index = training.shape[1] - 1

        # clean test data
        test_anon = np.delete(test, class_index, 1)

        # obtain classifications for test partition
        x = classifier(training, test_anon, args[0], args[1])
        
        # check classifications against actual classifications
        matches = [i for i in range(len(x)) if x[i] == test[i][class_index]]

        total += len(matches)
 
    return total/data.shape[0]


def KNNclassifier(training, test, k, dist_type):
    '''
    Classifier funciton that takes in training data and its lables, and
    using nearest neighbor, calculates the most probable lable for the
    samples in the test data set.
    '''

    class_index = training.shape[1] - 1
    
    # calculates distances matrix
    training_anon = np.delete(training, class_index, 1)
    dist = distance.cdist(test, training_anon , dist_type)
        
    # calculates lowest point/label and adds it to classification list
    classifiers = list()
    for i in range(dist.shape[0]):    
        lowest_dist_indicies = list(np.argsort(dist[i])[0:k])
        k_neighbors = [training[x][class_index] for x in lowest_dist_indicies]        
        classifiers.append(mode(k_neighbors))
        
    return np.array(classifiers)