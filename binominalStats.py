from scipy.stats import binom_test
from statsmodels.stats.proportion import proportion_confint,  proportions_ztest, proportions_chisquare
import numpy as np
import scipy.stats as stats
class binominalStats:
    def __init__(self, heads, total_flips, alpha=0.05, alternative='two-sided'):
        self.heads = heads
        self.total_flips = total_flips
        self.alpha = alpha
        self.alternative = alternative

    def binomial_test(self, base_rate=0.5):
        """
        Perform a binomial test to check if a coin is fair.

        Args:
            heads (int): Number of heads observed.
            total_flips (int): Total number of coin flips.
            alpha (float, optional): Significance level. Defaults to 0.05.
            alternative (str, optional): Type of test to perform ('two-sided', 'greater', or 'less'). Defaults to 'two-sided'.

        Returns:
            float: p-value of the test.
        """
        # Null hypothesis: the coin is fair (p = 0.5)
        return binom_test(self.heads, self.total_flips, p=base_rate, alternative=self.alternative)

    def exact_CI(self):
        """
        Calculate the exact confidence interval of a proportion where there is a wide range in the sample size or the proportion.

        This method avoids the assumption that data are normally distributed. The sample size and proportion are described by a beta distribution.

        Args:
            heads (int): Number of heads observed.
            total_flips (int): Total number of coin flips.
            alpha (float, optional): Confidence level. Defaults to 0.95.

        Returns:
            dict: The proportion with the lower and upper confidence intervals.
        """
        x = float(self.heads)
        N = float(self.total_flips)
        p = round((x/N)*100, 2)

        lower, upper = proportion_confint(count=x, nobs=N, alpha=self.alpha, method='wilson')
        lower_limit = max(0, round(lower * 100, 4))
        upper_limit = min(100, round(upper * 100, 4))

        result = {'Proportion': p, 'Lower CI': lower_limit, 'Upper CI': upper_limit}

        return result


class MultipleBinominalTests:
    """
    A class to perform multiple binominal tests for comparing the fairness of multiple categories.
    """

    def __init__(self, data_a,data_b,func, alternative = 'two-sided'):
        """
        Initialize the MultipleBinominalTests class with a variable number of categories.

        Args:
        *categories (numpy.ndarray): Observed frequencies for each category (format: [count1, count2]). No totals!
        """
        self.data_a = data_a
        self.data_b = data_b
        self.func = func
        self.alternative = alternative

    def main(self):
        """
        Perform the Chi-square test for independence and return the p-value.

        Args:
        alpha (float): Significance level for the test, default is 0.05.

        Returns:
        float: The p-value of the test.
        """
        # Create the contingency table
        
        counts       = np.array((self.data_a[0], self.data_b[0]))
        observations = np.array((np.sum(self.data_a), np.sum(self.data_b)))
        if self.func == 'ztest':
            p_value = proportions_ztest(count=counts, nobs=observations, alternative=self.alternative)[1]
        elif self.func == 'chi2':
            p_value = proportions_chisquare(count=counts, nobs=observations)[1]
        else:
            raise ValueError(f'binominalStats:MultipleBinominalTests: Unknown test function {self.func}')    
        
        if np.isnan(p_value):
            p_value =1
            print('binominalStats:MultipleBinominalTests: p-value is nan, either because samples are identical or nan in data. p-value set to 1')
        return p_value

    def test_result(self, alpha=0.05):
        """
        Determine if the categories have significantly different fairness based on the p-value and alpha.

        Args:
        alpha (float): Significance level for the test, default is 0.05.

        Returns:
        bool: True if the categories have significantly different fairness, False otherwise.
        """
        p_value = self.perform_test(alpha)

        # Compare the p-value to the significance level
        if p_value < alpha:
            return True
        else:
            return False
