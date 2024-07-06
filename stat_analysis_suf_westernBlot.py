import multiGroupTest,dataIO
import matplotlib.pyplot as plt
import binominalStats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
Script: stat_analysis_GargEtAl_2023.py
Written by: Bart R.H. Geurten
Date: 21. Nov 2023
Purpose: Statistical analysis of the immunohisto analysis cell cultures for the Garg et al. 2023B article.

This script is used for the statistical analysis of the immunohisto analysis cell cultures in the Garg et al. 2023B article.
It aims to test whether there is a difference in mitochondria presence between wildtype and mutant zebrafish. The analysis
employs multiple chi-square tests, which are corrected with the Benjamini Hochberg False Discovery Rate (FDR).

The multiGroupTest class is used to perform multiple statistical tests on the data, and the dataIO class is used to read
and manipulate the data.

The file_path variable contains a list of file paths to the csv files containing the original data. The script iterates over
the file paths, reads the data, creates a multiGroupTest object, runs the statistical tests, and saves the results to a csv file.

The statistical test used is a chi-square test to test for differences in the presence of mitochondria between wildtype and
mutant zebrafish. All p-values are corrected with the Benjamini Hochberg FDR detection routine.
'''

tag= "Westernblot_suf_brain"

# file_path contains a list of file paths to cs
# v files containing the original data
file_path = f'./Data/{tag}.csv'
df = pd.read_csv(file_path)
df['id'] = df['genotype'] + '_' + df['sex']

# data combinations to be tested

f = sns.barplot(
    data=df.loc[df['sex']== 'male',:], x="genotype", y="expression",
    hue="sex"
)
sns.stripplot(
    data=df.loc[df['sex']== 'male',:], x="genotype", y="expression", 
    jitter=True, color=[0.3, 0.3, 0.3] 
)

plt.gcf().savefig(f'./figures/{tag}_male.svg')



save_file = f'./stats/{tag}_IndepedentT_FDR_BH.csv'
# Read the data from the current file and convert it to a long table format
# Create a multiGroupTest object and set the data, group and test parameters
mgt = multiGroupTest.multiGroupTest(df["expression"],df['id'],'hypo:IndependentT','all',
                                    combination_set=[('+/+_male','+/-_male'),('+/+_male','-/-_male'),('+/-_male','-/-_male')])
# Run the main method of the multiGroupTest object
result_df = mgt.main()
# Save the result to a csv file
result_df.to_csv(save_file, index=False)


# ANOVA 
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

save_file = f'./stats/{tag}ANOVA_Tukey.csv'
# Read the data from the current file and convert it to a long table format
# Create a multiGroupTest object and set the data, group and test parameters
model = ols('expression ~ C(id)', data=df).fit()

anova_table = sm.stats.anova_lm(model, typ=2)
# Perform Tukey's HSD test (assuming your DataFrame is still named `df`)
tukey_results = pairwise_tukeyhsd(df['expression'], df['genotype'])

with open(save_file, "w") as file:

    # Write a header for the ANOVA table
    file.write("## ANOVA Results ##\n\n")

    # Write the ANOVA table as text
    file.write(anova_table.to_string())

    # Add some spacing
    file.write("\n\n")  # Add blank lines for separation

    # Write a header for the Tukey HSD results
    file.write("## Tukey HSD Results ##\n\n")

    # Write the Tukey results as text
    file.write(tukey_results.summary().as_text())

