Name: Cooper Bates
UNI: cbb2153


HW6_2 Nearest Neighbor

Description:
	The nn.py consists of three functions. The two prescribed ones are KNNclassifer and n_validator. KNNclassifier takes in two data sets as numpy arrays, one training set and one test set, as well as a k value and distance type to determine the k nearest neighbors to each data point in the test matrix. It then take the mode of these k nearest neighbors to determine the “best guess” classifier tag for that specific data point in the test matrix. It then appends this tag to an array containing the “best guesses” of classifying tags the computer extrapolates for all sample points in the test data set given the nearest neighbor approach.

	n_validator takes in several arguments, and essentially works to give a reading on the accuracy of the classifier function object inputed as an argument. It does this by first looking at a total data set as well as a partitioning index p inputed as an argument. Using a for loop, it calculates the most even way of breaking the total data set into p parts and then tests the classifier on the data partitions for all the possible combinations with the partitions created.

	partition_data is simply a helper function for n_validator.

	For nn_tester.py module, several functions combine to display the optimal k value, and k-distance_type combo, for accurately classifying test data based on training data. It does this by reading in the WDBC caner data from a text file, and finding the optimal k value of that, as well as by generating a synthetic data set and determining the optimal k value for a certain number of trials, then averaging all the trial results to reduce randomness.

Bugs:
	No bugs currently, that I have observed.

Additional Comments:
	I have submitted along with this assignment the file ‘wdbc.data.txt’, which is the Wisconsin Breast Cancer Data as a text file in the format which my program reads it. It is just a modified format of the original downloaded wdbc.data. The modifications I performed to the data before the program was run are as follows:
	-Replaced all instances of ‘M’ with ‘0’ and all instances of ’B’ with ‘1’
	-Replaced all ‘,’ with ‘ ‘

	The trial number for the optimal k value with synthetic data is set to 100 as to assignment specification, however since it is a feature designed to reduce randomness, the higher it is set the more consistent the optimal k value will be for consecutive executions of the test code.

	The program as a whole will take a bit of time to run, but will take less than a minute.