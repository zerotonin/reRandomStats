from scipy.stats import mannwhitneyu, kruskal, chisquare, kstest, mood, kruskal, ttest_ind, ranksums

class HypothesisTests:
    """
    A class for conducting various hypothesis tests on two samples.

    Note that this class is a wrapper around existing functions from the scipy.stats library.
    
    Available tests:
    
    - Mann-Whitney U test: a non-parametric test to compare two independent groups. It is suitable for 
      ordinal or continuous data that are not normally distributed. Limitation: assumes that the two 
      groups have the same shape and variability.
    
    - Kruskal-Wallis test: a non-parametric test to compare more than two independent groups. It is suitable 
      for ordinal or continuous data that are not normally distributed. Limitation: assumes that the groups 
      have the same shape and variability.
    
    - Chi-square test: a non-parametric test to test for independence between two categorical variables. It is 
      suitable for frequency data. Limitation: requires that the expected counts are greater than 5 for 
      each cell in the contingency table.
    
    - Kolmogorov-Smirnov test: a non-parametric test to compare the distributions of two independent samples. 
      It is suitable for continuous data that are not normally distributed. Limitation: sensitive to 
      differences in location, scale, and shape.
    
    - Mood's Median test: a non-parametric test to compare the medians of two independent groups. It is 
      suitable for ordinal or continuous data that are not normally distributed. Limitation: assumes that 
      the two groups have the same shape.
    
    - Wilcoxon Ranksum Test: a non-parametric test to compare two independent groups. It is suitable for 
      ordinal or continuous data that are not normally distributed. Limitation: assumes that the two 
      groups have the same shape and variability.
    
    - Independent t-test: a parametric test to compare the means of two independent groups. It is suitable 
      for continuous data that are normally distributed. Limitation: assumes that the two groups have the 
      same variance.
    
    """

    def __init__(self, data_a, data_b, func, alternative='two-sided'):
        """
        Initializes the HypothesisTests object.

        :param data_a: First sample data.
        :type data_a: list, numpy array, or pandas series
        :param data_b: Second sample data.
        :type data_b: list, numpy array, or pandas series
        :param func: Name of the hypothesis test to be performed. Currently supports 'MannWithneyU', 'KruskalWallis',
        'ChiSquare', 'Kolmogorov', 'MoodMedian', and 'IndependentT'.
        :type func: str
        :param alternative: Whether to use a one-sided or two-sided test. Default is 'two-sided'.
        :type alternative: str
        """
        self.data_a = data_a
        self.data_b = data_b
        self.func = func
        self.alternative = alternative

    def main(self):
        """
        Conducts the hypothesis test and returns the p-value.

        :return: The p-value of the hypothesis test.
        :rtype: float
        """
        if self.func == 'MannWhitneyU':
            # Mann-Whitney U test
            p_value = mannwhitneyu(self.data_a, self.data_b, alternative=self.alternative)[1]
        elif self.func == 'KruskalWallis':
            # Kruskal-Wallis test
            p_value = kruskal(self.data_a, self.data_b)[1]
        elif self.func == 'ChiSquare':
            # Chi-square test
            _, p_value, _, _ = chisquare(self.data_a, self.data_b)
        elif self.func == 'WilcoxonRankSum':
            _,p_value = ranksums(self.data_a, self.data_b, alternative=self.alternative)
        elif self.func == 'Kolmogorov':
            # Kolmogorov-Smirnov test
            _, p_value = kstest(self.data_a, self.data_b)
        elif self.func == 'MoodMedian':
            # Mood's median test
            p_value = mood(self.data_a, self.data_b, alternative=self.alternative)[1]
        elif self.func == 'IndependentT':
            # Independent t-test
            _, p_value = ttest_ind(self.data_a, self.data_b)
        else:
            raise ValueError(f'HypothesisTests:HypothesisTests: Unknown test function {self.func}')

        return p_value