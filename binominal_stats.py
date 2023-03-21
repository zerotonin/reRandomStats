from scipy.stats import beta
import scipy.stats as stats

def binomial_test(heads, total_flips, alpha=0.05,alternative='two-sided'):
    # Null hypothesis: the coin is fair (p = 0.5)
    return stats.binom_test(heads, total_flips, p=0.5, alternative=alternative)
    
    
# Calculate CI and p-Value for binominal distribution
def exact_CI(x, N, alpha=0.95):
    """
    Calculate the exact confidence interval of a proportion 
    where there is a wide range in the sample size or the proportion.

    This method avoids the assumption that data are normally distributed. The sample size
    and proportion are desctibed by a beta distribution.

    Parameters
    ----------

    x: the number of cases from which the proportion is calulated as a positive integer.

    N: the sample size as a positive integer.

    alpha : set at 0.95 for 95% confidence intervals.

    Returns
    -------
    The proportion with the lower and upper confidence intervals as a dict.

    """
    x = float(x)
    N = float(N)
    p = round((x/N)*100,2)

    intervals = [round(i,4)*100 for i in beta.interval(alpha,x,N-x+1)]
    intervals.insert(0,p)

    result = {'Proportion': intervals[0], 'Lower CI': intervals[1], 'Upper CI': intervals[2]}

    return result
