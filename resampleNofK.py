import random
from itertools import combinations

class getNofK:
    def __init__(self,dataSetA,dataSetB,combinationN, mode='resampling', maxLenPossible4Perms=20,resamplingN = 20000):
        self.dataSetA = list(dataSetA)
        self.dataSetB = list(dataSetB)
        self.combinationN = combinationN
        self.maxLenPossible4Perms = maxLenPossible4Perms
        self.resamplingN = resamplingN
        self.mode = mode

        self.combinedData = self.dataSetA + self.dataSetB
        self.combinedLen = self.getCombiLen()
        self.shortLen = self.getSmallSetN()

        if self.combinationN == 'all':
            self.mode = 'combinations'
        else:
            self.mode = 'resampling'

        if self.mode == 'combinations' and self.maxLenPossible4Perms < self.shortLen:
            self.mode = 'resampling'
            self.combinationN = self.resamplingN
        
    def getCombiLen(self):
        self.lenA = self.getDataLen(self.dataSetA)
        self.lenB = self.getDataLen(self.dataSetB)
        return self.lenA +self.lenB
    
    def getSmallSetN(self):
        if self.lenB < self.lenA:
            return self.lenB
        else:
            return  self.lenA

    def getDataLen(self,data):
        return len(list(data))
    
    def getAllCombinations(self):
        return list(combinations(range(self.combinedLen),self.shortLen))

    def getRandomCombinations(self):
        combinations = set()
        allIndiceList = list(range(self.combinedLen))
        tries = 0
        desperation = False
        while (len(combinations) < self.resamplingN) and not desperation :
            combinations.add(tuple(sorted(random.sample(allIndiceList,self.shortLen))))
            tries += 1
            if tries > self.combinationsN*2:
                desperation = True
                print(f'getNofK: getRandomCombinations: Could not produce more than {len(combinations)} in {tries} tries. So I use those ')
        return [tuple(x) for x in combinations]

    def combineIndices(self):
        allIndiceSet = set(range(self.combinedLen))
        self.dataIndices = []
        for indicesA in self.combinations:
            indicesB =tuple(allIndiceSet-set(indicesA))
            self.dataIndices.append((indicesA,indicesB))

    def main(self):
        if self.mode == 'combinations':
            self.combinations = self.getAllCombinations()
        if self.mode =='resampling':
            self.combinations = self.getRandomCombinations()
        self.combineIndices()
        self.combinationN = len(self.combinations)
    
    def getShuffeldSet(self,combiI):
        shuffleSetA = [self.combinedData[i] for i in self.dataIndices[combiI][0]]
        shuffleSetB = [self.combinedData[i] for i in self.dataIndices[combiI][1]]
        return shuffleSetA,shuffleSetB
