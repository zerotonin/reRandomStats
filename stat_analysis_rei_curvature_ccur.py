import multiGroupTest,dataIO
import matplotlib.pyplot as plt
import seaborn as sns
import binominalStats
import pandas as pd
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
tag= "rei_curvature_ccur"
# Assign the current file path, data subset, and save file name
file = f'./Data/rei_curvature_c_data.csv'
df = pd.read_csv(file)
df['id'] = df.sex + df.genotype

# PLOTS
import seaborn as sns
for parameter in ['median_curv_amp', 'mean_curv_amp', 'max_curv_amp']:
    f= plt.figure()
    sns.boxplot(x="genotype", y=parameter, order=['rei-INT', 'rei-HT', 'rei-HM'],
            hue="sex",hue_order=['M','F'],data=df)
    plt.savefig('./figures/'+f'rei_curvature_ccur--{parameter}.svg'.replace(' ','_').replace('/','_per_'))

plt.show()


#STATS

for parameter in ['median_curv_amp', 'mean_curv_amp', 'max_curv_amp']:
    

    combi_set = [('Mrei-INT','Mrei-HT'),('Mrei-INT','Mrei-HM'),('Mrei-HT','Mrei-HM'),
                ('Frei-INT','Frei-HT'),('Frei-INT','Frei-HM'),('Frei-HT','Frei-HM')]


    save_file = './stats/'+f'rei_curvature_ccur--{parameter}_FisherMean_FDR_BH.csv'.replace(' ','_').replace('/','_per_').replace(',','')
    mgt = multiGroupTest.multiGroupTest(df[parameter].to_list(),df['id'].to_list(),'Fisher:meanDiff',20000,
                                        combination_set=combi_set,correction_type='fdr_tsbh')
    # Run the main method of the multiGroupTest object
    result_df = mgt.main()
    # Save the result to a csv file
    result_df.to_csv(save_file, index=False)

            