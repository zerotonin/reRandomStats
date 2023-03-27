from scipy.stats import binom_test
from statsmodels.stats.proportion import proportion_confint
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

    def __init__(self, *categories):
        """
        Initialize the MultipleBinominalTests class with a variable number of categories.

        Args:
        *categories (numpy.ndarray): Observed frequencies for each category (format: [count1, count2]).
        """
        self.categories = categories

    def perform_test(self, alpha=0.05):
        """
        Perform the Chi-square test for independence and return the p-value.

        Args:
        alpha (float): Significance level for the test, default is 0.05.

        Returns:
        float: The p-value of the test.
        """
        # Create the contingency table
        contingency_table = np.vstack(self.categories)

        # Perform the Chi-square test for independence
        chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)

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
