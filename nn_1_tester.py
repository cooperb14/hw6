'''
Name: Cooper Bates
UNI: cbb2153

Tester file for the nn.py module, testing it with both the Wisconsin data
and the synthetically generated data
'''

import numpy as np
import nn_1

def main():
    
    # read in cancer data from text file
    with open('wdbc.data.txt') as file:
        cancer_data = file.readlines()
        
    # modifies cancer data to put it into proper format
    cancer_data = np.array([i.split() for i in cancer_data])
    class_vector = cancer_data[:,1]
    class_vector.shape = (cancer_data.shape[0],1)
    cancer_data = np.delete(cancer_data.astype('float16'), (0, 1), 1)
    cancer_data = np.hstack((cancer_data, class_vector))
    
    # prints 5 fold n_validator with cancer data and synthetic data
    synthetic_result = nn_1.n_validator(cancer_data, 5, nn_1.NNclassifier)
    print('5 fold validation of synthetic data: {}'.format(synthetic_result))
    
    print(' ')
    
    real_result = nn_1.n_validator(nn_1.synthetic_data(), 5, nn_1.NNclassifier)
    print('5 fold validation of real data: {}'.format(real_result))

    
main()

