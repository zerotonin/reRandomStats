import numpy as np
from resampleNofK import  getNofK

class FisherResamplingTest:

    def __init__(self,dataA,dataB,func):
        self.dataA = dataA
        self.dataB = dataB
        self.func  = func
        self.nOk   = getNofK(dataA,dataB,'all')
        self.nOk.main() 
        self.resampleN = self.nOk.combinationN

    
    def main(self):
        self.originalTestResult = self.calculateTest(self.dataA,self.dataB)
        self.shuffeldResults = sorted(self.bootStrapper()) # we sort to find the original in the next step
        self.indexOrigInShuffeld = self.getIndexOfTheMemberWithClosestsValue(self.shuffeldResults,self.originalTestResult)
        self.indexNormed = self.indexOrigInShuffeld/self.resampleN
        # the resulting bootstrapped distribution is parametric and therefore it does not matter if we are 0->1 or 1<-0
        if self.indexNormed > 0.5:
            self.indexNormed = abs(self.indexNormed-1)
        
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
            return self.calculateTest_medanDifferences(dataA,dataB)
        
        else:
            raise ValueError(f'FisherResamplingTest: calculateTest: the testType {self.func} is not implemented')

    
    def calculateTest_medianDifferences(self,dataA,dataB):
        return np.median(np.array(dataA)) - np.median(np.array(dataB))

    def calculateTest_meanDifferences(self,dataA,dataB):
        return np.mean(np.array(dataA)) - np.mean(np.array(dataB))





dataSetA = [1,223,237,3.56,.2500,4,.365,304]
dataSetB = [12,22,33,66,221,147,339,21,6,34.59]
frt = FisherResamplingTest(dataSetA,dataSetB,'medianDiff')
frt.main()


#df = pd.read_hdf('/home/bgeurten/PyProjects/dallas-dlc-seperate-multi-animal-analysis/BjoernDataMedianCILongInd4BoxPlotBest.h5',key='df')
#print(df)
