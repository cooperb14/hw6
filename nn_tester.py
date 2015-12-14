'''
Name: Cooper Bates
UNI: cbb2153

Tester file for the nn.py module, testing it with both the Wisconsin data
and the synthetically generated data
'''
import numpy as np
import nn


def synthetic_data():
    '''
    Generates a synthetic data set of 600 samples classified into
    two categories
    '''    
    
    #means
    m1 = [2.5, 3.5]
    m2 = [.5, 1]
    
    #covariance
    cov1 = [[1,1],[1,4.5]]
    cov2 = [[2,0],[0,1]]
    
    
    #class 1 created
    x1 = np.random.multivariate_normal(m1,cov1,300)
    labels = np.zeros(300)
    labels.shape = (300,1)
    x1_labled = np.hstack((x1, labels))
    
    #class 2 created
    x2 = np.random.multivariate_normal(m2,cov2,300)
    labels = np.ones(300)
    labels.shape = (300,1)
    x2_labled = np.hstack((x2, labels))
    
    # total data set created and shuffled
    total_data = np.vstack((x1_labled, x2_labled))
    np.random.shuffle(total_data)
    
    return total_data


def best_k(data_set, trials, dist_type):
    '''
    Determines the best k value for labeling test data given a 
    specific distance type
    '''
    
    k = 1
    val = .1
    last_val = 0
    
    # test all k values until success rate is optimized
    while(val > last_val):
        summation = []
        
        # tests k value for a number of trials to reduce randomness factor
        for i in range(trials):
            x = nn.n_validator(data_set, 5, nn.KNNclassifier, k, dist_type)
            summation.append(x)
        last_val = val
        val = sum(summation)/trials
        
        # only test odd values of k
        k += 2
        
    return (k-2), val


def main():
    '''
    Tester function that determines optimal values of K for both the WDBC
    cancer data set as well as a synthetically generated set
    '''
    
    # set 3 distance types to be used and intialize storage arrays
    dist_types = 'euclidean', 'cityblock', 'yule'
    vals = list()
    k_s = list()
    
    # read in cancer data from text file
    with open('wdbc.data.txt') as file:
        cancer_data = file.readlines()

    # modifies cancer data to put it into proper format
    cancer_data = np.array([i.split() for i in cancer_data])
    class_vector = cancer_data[:,1]
    class_vector.shape = (cancer_data.shape[0],1)
    cancer_data = np.delete(cancer_data.astype('float16'), (0, 1), 1)
    cancer_data = np.hstack((cancer_data, class_vector))

    # determine optimal k for WDBC data
    print('For the WDBC data set:')
    for d_type in dist_types:
        value = best_k(cancer_data, 1, d_type)
        k_s.append(value[0])
        vals.append(value[1])
    
    # Print optimal k values for each distance type
    print('Optimal K value for Euclidean distance is {}'.format(k_s[0]))
    print('Optimal K value for Cityblock distance is {}'.format(k_s[1]))
    print('Optimal K value for Yule distance is {}'.format(k_s[2]))
    
    # Determine and Print best k and distance type combination
    best_index = np.argsort(vals)[2]
    print('Optimal combo is {} distance with a k value {}'\
    .format(dist_types[best_index], k_s[best_index]))
    
    # create a line break between different data sets
    print(' ')
    vals = list()
    k_s = list()
    
    # create synthetic data set
    data = synthetic_data()
    
    # determine optimal k for synthetic data set
    for d_type in dist_types:
        value = best_k(data, 100, d_type)
        k_s.append(value[0])
        vals.append(value[1])
    
    # Print optimal k values for each distance type
    print('Optimal K value for Euclidean distance is {}'.format(k_s[0]))
    print('Optimal K value for Cityblock distance is {}'.format(k_s[1]))
    print('Optimal K value for Yule distance is {}'.format(k_s[2]))
    
    # Determine and Print best k and distance type combination
    best_index = np.argsort(vals)[2]
    print('Optimal combo is {} distance with a k value {}'\
    .format(dist_types[best_index], k_s[best_index]))


main()

