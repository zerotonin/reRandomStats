import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import combinations
from FisherResampling import FisherResamplingTest
import statsmodels.api as sm
from tqdm import tqdm

class multiGroupTest:

    def __init__(self,data,group,test,correctionType='fdr_bh'):
        """[summary]

        :param data: [description]
        :type data: [type]
        :param group: [description]
        :type group: [type]
        :param test: [description]
        :type test: [type]
        :param correctionType: [description]
        :type correctionType: [type]
            Method used for testing and adjustment of pvalues. Can be either the full name or initial letters. Available methods are:

            bonferroni : one-step correction

            sidak : one-step correction

            holm-sidak : step down method using Sidak adjustments

            holm : step-down method using Bonferroni adjustments

            simes-hochberg : step-up method (independent)

            hommel : closed method based on Simes tests (non-negative)

            fdr_bh : Benjamini/Hochberg (non-negative)

            fdr_by : Benjamini/Yekutieli (negative)

            fdr_tsbh : two stage fdr correction (non-negative)

            fdr_tsbky : two stage fdr correction (non-negative)
        """        
        self.data = data
        self.group =group
        self.test = test
        self.correctionType = correctionType
    
    
    def rearrangeData(self):
        self.groupNames  = list()
        self.groupedData = list()
        for name in set(self.group):
            indices = [i for i, x in enumerate(self.group) if x == name]
            self.groupedData.append([self.data[i] for i in indices])
            self.groupNames.append(name)
        self.groupedData = tuple(self.groupedData)
        self.groupNames  = tuple(self.groupNames)

    def main(self):
        self.rearrangeData()
        self.groupCombis = self.getCombinations()                                 
        self.pValues  = self.runTests()
        self.sig,self.pValsCorr,alphacSidak,alphacBonf = sm.stats.multipletests(self.pValues, alpha=0.05, method=self.correctionType)
        self.df = self.createOutput()
        return self.df

    def createOutput(self):
        firstDataSet = [self.groupNames[x[0]] for x in self.groupCombis]
        secondDataSet = [self.groupNames[x[1]] for x in self.groupCombis]
        return pd.DataFrame({'groupA':firstDataSet,'groupB':secondDataSet,'p value':self.pValues,'p value corrected':self.pValsCorr,'h':self.sig})

    def getCombinations(self):
        return list(combinations(range(len(self.groupNames)),2))

    
    def runTests(self):
        # get all dataSet combinations
        pValues = list()
        self.testObj = self.choose_test()
        for groupIndices in tqdm(self.groupCombis,'testing group combinations'):
            self.testObj.dataA = self.groupedData[groupIndices[0]]
            self.testObj.dataB = self.groupedData[groupIndices[1]]
            pVal = self.testObj.main()
            pValues.append(pVal)
        return pValues

    def choose_test(self):
        """
        This function chooses a test to use based on the test family and specific test specified.
        It splits the test string into test family and specific test, then checks if the test family is 'Fisher' or ...
        If the test family is 'Fisher', it returns a FisherResamplingTest object with specific test.
        Else, it raises a ValueError if the test family is not implemented
        
        Args:
        self: The instance of the class
        
        Returns:
        A test object of the following options
        FisherResamplingTest: A Roland Fisher resampling test object from FisherResamplin.py
        
        Raises:
        ValueError: if the test family is not implemented
        
        Example:
        
        test_instance = MyTestClass()
        test_instance.test = 'Fisher:test1'
        chosen_test = test_instance.choose_test()
        print(chosen_test) # FisherResamplingTest([],[], 'test1')
        """
        # split the test string into test family and specific test
        test_family, specific_test = self.test.split(':')

        # if the test family is 'Fisher', return a FisherResamplingTest object with specific test
        if test_family == 'Fisher':
            return FisherResamplingTest([],[], specific_test)
        # elif test_family == 'MannWhitneyU':
        #     return stats.mannwhitneyu([],[])
        # if the test family is not implemented, raise a ValueError
        else:
            raise ValueError(f'multiTestAnalysis: chooseTest: the test family {test_family} is not implemented')