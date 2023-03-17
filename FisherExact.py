import numpy as np
from scipy.stats import fisher_exact


class FisherExactTest:
    def __init__(self, data_a, data_b, alternative='two-sided'):
        """
        Initializes the FisherExactTest class.

        Args:
            data_a (tuple): The first data set with option A (e.g., alive) and option B (e.g., dead).
            data_b (tuple): The second data set with both options; both tuples are always sorted (A, B).
            alternative (str, optional): Defines the alternative hypothesis. 
                                         The following options are available (default is 'two-sided'):
                                         - 'two-sided': The odds ratio of the underlying population is not one.
                                         - 'less': The odds ratio of the underlying population is less than one.
                                         - 'greater': The odds ratio of the underlying population is greater than one.

        See the Notes for more details.
        """
        self.data_a = data_a
        self.data_b = data_b
        self.alternative = alternative
        self.p_value = None

    def main(self):
        """
        Performs a Fisher's exact test to compare two data sets.

        The method first constructs a 2x2 table using the input data sets. Then, it calculates the Fisher's
        exact test using the `fisher_exact` function from the `scipy.stats` module. Finally, it extracts and
        returns the p-value.

        Returns:
            float: The p-value of the Fisher's exact test.
        """
        # Construct a 2x2 table using the input data sets
        table = np.array([list(self.data_a), list(self.data_b)])

        # Calculate the Fisher's exact test using the `fisher_exact` function
        result = fisher_exact(table, alternative=self.alternative)

        # Extract the p-value from the result
        self.p_value = result[1]

        return self.p_value
#test = FisherExactTest((8,2),(1,5))
#test.main()