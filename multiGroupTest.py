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
        self.testObj = self.chooseTest()
        for groupIndices in tqdm(self.groupCombis,'testing group combinations'):
            self.testObj.dataA = self.groupedData[groupIndices[0]]
            self.testObj.dataB = self.groupedData[groupIndices[1]]
            pVal = self.testObj.main()
            pValues.append(pVal)
        return pValues
    
    def chooseTest(self):
        testFamily,specificTest = self.test.split(':')

        if testFamily == 'Fisher':
            return FisherResamplingTest([],[], specificTest)
        #elif testFamily == 'MannWhitneyU':
        #    return stats.mannwhitneyu([],[])
        else:
            raise ValueError(f'multiTestAnalysis: chooseTest: the test family {testFamily} is not implemented')
