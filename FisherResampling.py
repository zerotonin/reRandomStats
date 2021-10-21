import numpy as np
from resampleNofK import  getNofK

class FisherResamplingTest:

    def __init__(self,dataA,dataB,func):
        self.dataA = dataA
        self.dataB = dataB
        self.func  = func



    def getShuffeldIndices(self):
        self.nOk   = getNofK(self.dataA,self.dataB,'all')
        self.nOk.main() 
        self.resampleN = self.nOk.combinationN

    
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
            dataShuffleA, dataShuffelB = self.nOk.getShuffeldSet(i)
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


