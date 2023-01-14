import pandas as pd
from itertools import combinations
from FisherResampling import FisherResamplingTest
import statsmodels.api as sm
from tqdm import tqdm

class multiGroupTest:
    """
    A class for running multiple statistical tests on grouped data and applying multiple test correction.
    """

    def __init__(self,data,group,test,correction_type='fdr_bh'):
        """
        Initialize a multiGroupTest object.
        
        Parameters:
            data (list): A list of numerical values representing the data set to be tested.
            group (list): A list of strings representing the group names corresponding to each element of the data set.
            test (str): A string representing the statistical test to be used. The format should be test-family:test-name, e.g 'ttest:ind'
            correction_type (str, optional): The method used for testing and adjustment of pvalues. Can be either the full name or initial letters. Default is 'fdr_bh'. Available methods are:
                
                'bonferroni' : one-step correction

                'sidak' : one-step correction

                'holm-sidak' : step down method using Sidak adjustments

                'holm' : step-down method using Bonferroni adjustments

                'simes-hochberg' : step-up method (independent)

                'hommel' : closed method based on Simes tests (non-negative)

                'fdr_bh' : Benjamini/Hochberg (non-negative)

                'fdr_by' : Benjamini/Yekutieli (negative)

                'fdr_tsbh' : two stage fdr correction (non-negative)

                'fdr_tsbky' : two stage fdr correction (non-negative)
        """        
        self.data  = data
        self.group = group
        self.test  = test
        self.correction_type = correction_type
    

    def rearrange_data(self):
        """
        This function rearranges the data by grouping the elements of the original data set by their corresponding group name.
        It first creates an empty list for the group names, and another for the grouped data.
        Then, it iterates over the unique group names, and for each name, finds the indices of the elements in the original data set that belong to that group.
        It then appends the corresponding elements of the original data set to the grouped data list, and the group name to the group names list.
        Finally, it converts the grouped data and group names lists to tuples.
        """
        self.group_names  = list()
        self.grouped_data = list()
        for name in set(self.group):
            # Using list comprehension to find the indices of elements in the original data set that belong to a specific group
            indices = [i for i, x in enumerate(self.group) if x == name]
            # Appending the corresponding elements of the original data set to the grouped data list
            self.grouped_data.append([self.data[i] for i in indices])
            self.group_names.append(name)
        self.grouped_data = tuple(self.grouped_data)
        self.group_names  = tuple(self.group_names)


    def main(self):
        """
        This is the main function of the class.
        It first calls the `rearrange_data()` function to group the data by their corresponding group names.
        Then it calls `get_combinations()` function to get all possible unique combinations of the group names.
        Next, it calls `run_tests()` to run the statistical tests on all the combinations of the data sets and return a list of p-values.
        It then applies multiple-test-correction on the p-values using `scipy.stats.multipletests` function.
        Next it calls `create_output()` to create a DataFrame containing the results of the statistical tests.
        Finally, it returns the DataFrame containing the test results.
        """
        self.rearrange_data()
        self.group_combinations = self.get_combinations()
        self.p_values = self.run_tests()
        self.sig, self.p_values_corrected, _, _ = sm.stats.multipletests(self.p_values, alpha=0.05, method=self.correction_type)
        self.df = self.create_output()
        return self.df


    def create_output(self):
        """
        This function creates a DataFrame containing the results of the statistical tests.
        It first creates lists for the names of the first and second data sets in each combination, 
        the corrected p-values, and the significance levels.
        Then, it uses list comprehension to create these lists of the data groups and significance 
        levels bby iterating over the `group_combinations` and `pValsCorr` attributes, respectively.
        Finally, it returns a DataFrame containing the lists as columns, along with the original p-values 
        and the significance of the test results.
        """
        first_data_set = [self.group_names[x[0]] for x in self.group_combinations]
        second_data_set = [self.group_names[x[1]] for x in self.group_combinations]
        sig_level = [self.get_significance_level(p) for p in self.p_values_corrected]
        return pd.DataFrame({'groupA': first_data_set, 'groupB': second_data_set, 'p value': self.p_values, 
        'p value corrected': self.p_values_corrected, 'h': self.sig, 'sig. level': sig_level})


    
    def get_combinations(self):
        """
        This function generates a list of all possible unique tuples, each containing 2 indices out of the length of group names.
        
        Returns:
            list: A list of tuples containing all possible unique combinations of 2 indices out of the length of group names
        """
        # Using python's built-in combinations function from the itertools module to generate a list of all possible unique tuples
        return list(combinations(range(len(self.group_names)), 2))

    
    def get_significance_level(self,p_value):
        """
        This function returns a string indicating the level of significance based on a given p-value.
        The string returned is one of 'n.s.', '*', '**', or '***', corresponding to p-values greater than 0.05, between 0.01 and 0.05, between 0.001 and 0.01, and less than 0.001, respectively.
        
        Args:
            p_value (float): The p-value to determine the level of significance
        
        Returns:
            str: The level of significance indicated by the p-value
        """ 
        if p_value > 0.05:
            return 'n.s.'
        if p_value >0.01:
            return '*'
        if p_value > 0.001:
            return '**'
        if p_value < 0.001:
            return '***'


    def run_tests(self):
        """
        This function runs statistical tests on all possible unique combinations of the data sets and returns a list of p-values.
        It first calls the `choose_test()` function to determine which test to run, and then iterates over all unique combinations of the data sets using the `groupCombis` attribute.
        For each combination, it assigns the corresponding data sets to the `data_a` and `data_b` attributes of the test object, and calls the `main()` function of the test object to get the p-value.
        The p-values are then collected in a list and returned.
        """
        p_values = list()
        self.test_obj = self.choose_test()
        for group_indices in tqdm(self.group_combinations, 'testing group combinations'):
            self.test_obj.data_a = self.grouped_data[group_indices[0]]
            self.test_obj.data_b = self.grouped_data[group_indices[1]]
            p_val = self.test_obj.main()
            p_values.append(p_val)
        return p_values


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