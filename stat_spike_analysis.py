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

tag= "counter_current"

# file_path contains a list of file paths to cs
# v files containing the original data
file_path = f'./Data/fishDataBase_cstart.csv'
df = pd.read_csv(file_path)
df['id'] = df.sex + df.genotype
df['gene'] = 'rei'
df.loc[df['genotype'].apply(lambda x: 'sufge1' in x), 'gene'] = 'sufge1'


fields = ['latency_to_m_cell', 'latency_to_others', 'm_cell_spikes',
       'median_spike_instFreq_Hz', 'other_spikes']

genes = set([x.split('-')[0] for x in df.genotype.unique()])
df['gene'] =  [x.split('-')[0] for x in df.genotype]

for gtype in genes:
    df_gen = df.loc[df.gene == gtype, :]
    for field in fields:
        plt.figure()
        sns.boxplot(data=df_gen, x='genotype', y=field, hue='sex', hue_order=['M', 'F'],
                    order=[f'{gtype}-INT', f'{gtype}-HT', f'{gtype}-HM'])
        # Set the y-axis to logarithmic scale
        plt.yscale('log')

        # Save the figure as an SVG file
        plt.savefig(f'./figures/ePhys_{gtype}_{field}.svg')

        # Close the current figure to avoid overlapping plots
        plt.close()

plt.show()


combi_rei    = [('Mrei-INT','Mrei-HT'),('Mrei-INT','Mrei-HM'),('Mrei-HT','Mrei-HM'),
                ('Frei-INT','Frei-HT'),('Frei-INT','Frei-HM'),('Frei-HT','Frei-HM')]
combi_sufge1 = [('Msufge1-INT','Msufge1-HT'),('Msufge1-INT','Msufge1-HM'),('Msufge1-HT','Msufge1-HM'),
                ('Fsufge1-INT','Fsufge1-HT'),('Fsufge1-INT','Fsufge1-HM'),('Fsufge1-HT','Fsufge1-HM')]
for gtype in genes:
    df_gen = df.loc[df.gene == gtype, :]
    if gtype == 'sufge1':
        combi_set = combi_sufge1
    else:
        combi_set = combi_rei

    for field in fields:
        df_temp = df_gen.loc[:,['id',field]]
        df_temp = df_temp.dropna()
        save_file = f'./stats/ePhys_{gtype}_{field}_FisherMean_FDR_BH.csv'
        # Read the data from the current file and convert it to a long table format
        # Create a multiGroupTest object and set the data, group and test parameters
        mgt = multiGroupTest.multiGroupTest(df_temp[field].to_list(),df_temp['id'].to_list(),'Fisher:meanDiff',10000,
                                            combination_set=combi_set,correction_type='fdr_tsbh')
        # Run the main method of the multiGroupTest object
        result_df = mgt.main()
        # Save the result to a csv file
        result_df.to_csv(save_file, index=False)




