import numpy as np
from resampleNofK import getNofK
class FisherResamplingTest:

    def __init__(self, data_a, data_b, func):
        """
            This class performs resampling statistics using the method of Ronald Fisher to compare two data sets.

        Args:
            data_a (list): the first data set for comparison
            data_b (list): the second data set for comparison
            func (function): the function to be used for comparison of the data sets
            max_n (int): the combined maximum length of both datasets, for which all n out of k pairings can be calculated. 
                         all n out of k are 9! combinations which are 362.880 pairtings. After this the system draws 100.000 pairings
        """
        self.data_a = data_a
        self.data_b = data_b
        self.func  = func

    def get_shuffled_indices(self):
        """
        This function generates shuffled indices for both data sets A and B. It uses the getNofK function to 
        calculate the number of elements in the combined data set and the number of elements in each of the 
        original data sets.
        """
        # get the number of elements in the combined data set and the number of elements in each of the original data sets
     
        self.n_of_k = getNofK(self.data_a, self.data_b, 'all')
        self.n_of_k.main() 
        self.resample_n = self.n_of_k.combination_n

    
    def main(self):
        """
        Performs a resampling test of Ronald Fisher to compare two data sets. 
        
        The method first gets shuffled indices of the data sets by calling the `get_shuffled_indices()` method. 
        Then it calculates the original difference between the two data sets by calling the `calculate_test()` method. 
        Next, it calculates the shuffled results by calling the `boot_strapper()` method and sorts the results. 
        It finds the original result in the shuffled results and normalizes it to the number of results produced. 
        Finally, it calculates the p-value and returns it.
        
        Returns:
            float: The p-value of the resampling test.
        """
        # Get shuffled indices of the data sets
        self.get_shuffled_indices()
        # Get the original difference between the data sets
        self.original_test_result = self.calculate_test(self.data_a, self.data_b)
        # Calculate shuffled results
        self.shuffled_results = sorted(self.bootstrap_resampling())
        # Find original result in shuffled results and normalize to the number of results produced
        self.index_of_original_in_shuffled = self.get_index_of_closest_value(self.shuffled_results, self.original_test_result)
        self.index_normalized = self.index_of_original_in_shuffled / self.resample_n
        # Calculate the p-value
        if self.index_normalized > 0.5:
            self.index_normalized = abs(self.index_normalized - 1)
        if self.index_normalized == 0.0:
            self.index_normalized = 1.0 / self.resample_n
        self.p_value = self.index_normalized * 2
        return self.p_value

    def get_index_of_closest_value(self, values_list, value_to_match):
        """
        This function takes a list of values and a target value, and returns the index of the element in the list that is closest to the target value.
        """
        return min(range(len(values_list)), key=lambda i: abs(values_list[i]-value_to_match))

    def bootstrap_resampling(self):
        """
        This function performs a bootstrap resampling of the data sets provided in the class initialization.
        It shuffles the data sets and calculates the test statistic for the shuffled data sets.
        It returns a list of test statistics for the resampled data sets.
        """

        # run tests
        resampled_results = []
        for i in range(self.resample_n):
            shuffled_data_set_a, shuffled_data_set_b = self.n_of_k.get_shuffled_set(i)
            resampled_results.append(self.calculate_test(shuffled_data_set_a, shuffled_data_set_b))
        return resampled_results


    
    def calculate_test(self, dataA, dataB):
        """
        This function is used to calculate the test statistic for the two data sets. The test statistic is determined by the value of the func attribute, which can be either 'medianDiff' or 'meanDiff'.
        Args:
            dataA (list): The first data set to be compared
            dataB (list): The second data set to be compared

        Returns:
            float: The calculated test statistic

        Raises:
            ValueError: If the value of the `func` attribute is not 'medianDiff' or 'meanDiff'
        """
        if self.func == 'medianDiff':
            return self.calculateTest_medianDifferences(dataA, dataB)
        elif self.func == 'meanDiff':
            return self.calculateTest_meanDifferences(dataA, dataB)
        else:
            raise ValueError(f'FisherResamplingTest: calculateTest: the testType {self.func} is not implemented')

    
    def calculateTest_medianDifferences(self,dataA,dataB):
        """
        This function calculates the test statistic for the two data sets as the difference in medians.

        Copy code
        Args:
            dataA (list): The first data set to be compared
            dataB (list): The second data set to be compared

        Returns:
            float: The calculated test statistic
        """
        return np.median(np.array(dataA)) - np.median(np.array(dataB))

    def calculateTest_meanDifferences(self,dataA,dataB):
        """
        This function calculates the test statistic for the two data sets as the difference in means.

        Copy code
        Args:
            dataA (list): The first data set to be compared
            dataB (list): The second data set to be compared

        Returns:
            float: The calculated test statistic
        """
        return np.mean(np.array(dataA)) - np.mean(np.array(dataB))





