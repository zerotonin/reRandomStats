import multiGroupTest,dataIO
import matplotlib.pyplot as plt
import binominalStats
import numpy as np
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


# file_path contains a list of file paths to csv files containing the original data
tag= "cell_culture_data_female"
#tag= "cell_culture_data_male"

# Assign the current file path, data subset, and save file name
file = f'./Data/{tag}.csv'
save_file = f'./stats/{tag}_FishersExactTest_FDR_BH.csv'

# data combinations to be tested
combinations = [('wt_c1', 'mut_c1'), ('wt_c2', 'mut_c2'), ('wt_c3','mut_c3'), ('wt_c4','mut_c4')] 


# Create a DataIO object
dio = dataIO.DataIO()  
# Read the data from the current file and convert it to a long table format
id_list,data=dio.wide_table_csv_to_long_table(file)
# Create a multiGroupTest object and set the data, group and test parameters
mgt = multiGroupTest.multiGroupTest(data,id_list,'Fisher:exact','set',combination_set=combinations)
# Run the main method of the multiGroupTest object
result_df = mgt.main()
# Save the result to a csv file
result_df.to_csv(save_file)

binoObj = binominalStats.binominalStats(0,0)

temp = np.array([data[0:8],data[8::]])
conf_int = list()
for i in range(temp.shape[1]):
    binoObj.heads = temp[0,i]
    binoObj.total_flips = temp[1,i]
    conf_int.append(binoObj.exact_CI())

import matplotlib.pyplot as plt

def plot_bar_with_error_bars(group1, group2, ci_lower1, ci_upper1, ci_lower2, ci_upper2, color1='teal', color2='orange',g_name1='wt',g_name2='mut'):
    labels = ['1', '2', '3', '4']
    x = range(len(labels))
    width = 0.4

    fig, ax = plt.subplots()
    
    # Calculate error bars for group 1
    error_bars1 = [[group1[i] - ci_lower1[i] for i in range(len(group1))], [ci_upper1[i] - group1[i] for i in range(len(group1))]]
    ax.bar([i - width/2 for i in x], group1, width, yerr=error_bars1, label=g_name1, color=color1)
    
    # Calculate error bars for group 2
    error_bars2 = [[group2[i] - ci_lower2[i] for i in range(len(group2))], [ci_upper2[i] - group2[i] for i in range(len(group2))]]
    ax.bar([i + width/2 for i in x], group2, width, yerr=error_bars2, label=g_name2, color=color2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()
    return fig,ax


fig,ax =plot_bar_with_error_bars([x['Proportion']for x in conf_int[0:4]], 
                                 [x['Proportion']for x in conf_int[4::]], 
                                 [x['Lower CI']for x in conf_int[0:4]], 
                                 [x['Upper CI']for x in conf_int[0:4]], 
                                 [x['Lower CI']for x in conf_int[4::]], 
                                 [x['Upper CI']for x in conf_int[4::]])

ax.set_xlabel('anatomical location')
ax.set_ylabel('cell positive, in percent')
fig.savefig(f'./figures/{tag}.svg')
