import numpy as np
from resampleNofK import  getNofK

class FisherResamplingTest:

    def __init__(self, dataA, dataB, func):
        """
        Initializes the class with two data sets and a function to be used for comparison.

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
        # do the shuffeling
        self.getShuffeldIndices()
        # get the original difference
        self.originalTestResult = self.calculateTest(self.dataA,self.dataB)
        # calculate the shuffeld results
        self.shuffeldResults = sorted(self.bootStrapper()) # we sort to find the original in the next step
        # find the original result and normalise to the number of results we produced
        self.indexOrigInShuffeld = self.getIndexOfTheMemberWithClosestsValue(self.shuffeldResults,self.originalTestResult)
        self.indexNormed = self.indexOrigInShuffeld/self.resampleN
        # the resulting bootstrapped distribution is parametric and therefore it does not matter if we are 0->1 or 1<-0
        if self.indexNormed > 0.5:
            self.indexNormed = abs(self.indexNormed-1)
        
        if self.indexNormed == 0.0:
            self.indexNormed = 1.0/self.resampleN

        self.pValue = self.indexNormed*2 # because this is a two sided test

        return self.pValue


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


