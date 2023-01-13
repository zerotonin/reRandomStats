import numpy as np
from resampleNofK import  getNofK

class FisherResamplingTest:

    def __init__(self, dataA, dataB, func):
        """
            This class performs resampling statistics using the method of Ronald Fisher to compare two data sets.

        Args:
            dataA (list): the first data set for comparison
            dataB (list): the second data set for comparison
            func (function): the function to be used for comparison of the data sets
        """
        self.dataA = dataA
        self.dataB = dataB
        self.func  = func

    def get_shuffled_indices(self):
        """
        This function generates shuffled indices for both data sets A and B. It uses the getNofK function to 
        calculate the number of elements in the combined data set and the number of elements in each of the 
        original data sets.
        """
        # get the number of elements in the combined data set and the number of elements in each of the original data sets
        self.n_of_k = getNofK(self.dataA, self.dataB, 'all')
        self.n_of_k.main() 
        self.resampleN = self.n_of_k.combinationN

    
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
        self.original_test_result = self.calculate_test(self.dataA, self.dataB)
        # Calculate shuffled results
        self.shuffled_results = sorted(self.boot_strapper())
        # Find original result in shuffled results and normalize to the number of results produced
        self.index_of_original_in_shuffled = self.get_index_of_member_with_closest_value(self.shuffled_results, self.original_test_result)
        self.index_normalized = self.index_of_original_in_shuffled / self.resample_n
        # Calculate the p-value
        if self.index_normalized > 0.5:
            self.index_normalized = abs(self.index_normalized - 1)
        if self.index_normalized == 0.0:
            self.index_normalized = 1.0 / self.resample_n
        self.p_value = self.index_normalized * 2
        return self.p_value


    def getIndexOfTheMemberWithClosestsValue(self,a_list,value2match): 
        return min(range(len(a_list)), key=lambda i: abs(a_list[i]-value2match))

    def bootStrapper(self):
        result = list()
        for i in range(self.resampleN):
            dataShuffleA, dataShuffelB = self.n_of_k.getShuffeldSet(i)
            result.append(self.calculateTest(dataShuffleA,dataShuffelB))
        return result

    
    def calculateTest(self,dataA,dataB):
        
        if self.func == 'medianDiff':
            return self.calculateTest_medianDifferences(dataA,dataB)
        
        elif self.func == 'meanDiff':
            return self.calculateTest_meanDifferences(dataA,dataB)
        
        else:
            raise ValueError(f'FisherResamplingTest: calculateTest: the testType {self.func} is not implemented')

    
    def calculateTest_medianDifferences(self,dataA,dataB):
        return np.median(np.array(dataA)) - np.median(np.array(dataB))

    def calculateTest_meanDifferences(self,dataA,dataB):
        return np.mean(np.array(dataA)) - np.mean(np.array(dataB))


