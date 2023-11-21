import multiGroupTest,dataIO
import matplotlib.pyplot as plt
import binominalStats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
Script: stat_analysis_suf_ephys.py
Written by: Bart R.H. Geurten
Date: 21. Nov 2023
Purpose: Testing the significance of differences in large spike abundance after an air stimulus in adult fish electrophysiological recordings.

This script performs a statistical analysis to determine whether there is a significant difference in the abundance of large spikes after an air
stimulus in electrophysiological recordings of adult fish.

The data for the analysis is acquired by running the 'run_sufge1_c-start_ana.py' script from the pyLACE project (https://github.com/zerotonin/pyLACE). 
In this process, the 'get_spike_mauthner_histogram' function is used with the normalization flag set to false. The script then summarizes all Mauthner 
spikes post-stimulus (index 110) and stores the data in the format data[sex][genotype][mauthner_histogram].

The script employs multiple chi-square tests, corrected with the Benjamini Hochberg False Discovery Rate (FDR), to assess the presence of mitochondria in wildtype
 vs. mutant zebrafish. It uses the multiGroupTest class for statistical testing and the dataIO class for reading data.

The 'file_path' variable lists the paths to CSV files containing original data. The script iterates over these paths, reads the data, performs statistical tests 
using multiGroupTest, and saves the results to CSV files. The primary statistical test is a chi-square test to check for differences in mitochondria presence 
between wildtype and mutant zebrafish, with all p-values corrected using the Benjamini Hochberg FDR routine.
'''

tag= "mauthner_large_spikes_post_stim"

# file_path contains a list of file paths to cs
# v files containing the original data
file_path = f'./Data/{tag}.csv'
df = pd.read_csv(file_path)

# data combinations to be tested

f = sns.barplot(
    data=df, x="genotype", y="large_spikes" , # sex-specific df.loc[df['sex']== 'male',:]
)
sns.stripplot(
    data=df, x="genotype", y="large_spikes", 
    jitter=True, color = [0.5, 0.5, 0.5]
)

plt.gcf().savefig(f'./figures/{tag}.svg')



save_file = f'./stats/{tag}_IndepedentT_FDR_BH.csv'
# Read the data from the current file and convert it to a long table format
# Create a multiGroupTest object and set the data, group and test parameters

set_combination = [('+/+',"+/-"),('+/+',"-/-"),('+/-',"-/-")]
mgt = multiGroupTest.multiGroupTest(df["large_spikes"],df['genotype'],'Fisher:sumDiff',500000, #'hypo:IndependentT','all',
                                    combination_set=set_combination)
# Run the main method of the multiGroupTest object
result_df = mgt.main()
# Save the result to a csv file
result_df.to_csv(save_file, index=False)




