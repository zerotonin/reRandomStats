from scipy.stats import beta, binom_test

class BinomialStats:
    def __init__(self, heads, total_flips, alpha=0.05, alternative='two-sided'):
        self.heads = heads
        self.total_flips = total_flips
        self.alpha = alpha
        self.alternative = alternative

    def binomial_test(self):
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
        return binom_test(self.heads, self.total_flips, p=0.5, alternative=self.alternative)

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

        intervals = [round(i, 4)*100 for i in beta.interval(self.alpha, x, N-x+1)]
        intervals.insert(0, p)

        result = {'Proportion': intervals[0], 'Lower CI': intervals[1], 'Upper CI': intervals[2]}

        return result
