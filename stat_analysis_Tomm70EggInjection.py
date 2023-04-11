import multiGroupTest,dataIO
import matplotlib.pyplot as plt
import binominalStats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
Script: stat_analysis_GargEtAl_2023B.py
Written by: Bart R.H. Geurten
Date: 18. Mar 2023
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

tag= "egg_Tomm70_injection"

# file_path contains a list of file paths to cs
# v files containing the original data
file_path = f'./Data/{tag}.csv'
df = pd.read_csv(file_path)
df_fish = df.groupby('fish_id', as_index=False).median()
df_fish[['genotype', 'injection']] = df.groupby('fish_id')[['genotype', 'injection']].first().reset_index(drop=True)
df_fish['id'] = df_fish['genotype']+" "+  df_fish['injection']

# data combinations to be tested

f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(
    data=df_fish, x="genotype", y="RawIntden_normatoMaxintensity",
    notch=False, showcaps=False,
    flierprops={"marker": "x"},
    hue="injection",
    )
plt.show()

f.savefig(f'./figures/{tag}_.svg')


save_file = f'./stats/{tag}_Fishers_MedianDiff_FDR_BH.csv'
# Read the data from the current file and convert it to a long table format
# Create a multiGroupTest object and set the data, group and test parameters
mgt = multiGroupTest.multiGroupTest(df_fish["Median"],df_fish['id'],'Fisher:medianDiff',20000)
# Run the main method of the multiGroupTest object
result_df = mgt.main()
# Save the result to a csv file
result_df.to_csv(save_file, index=False)




