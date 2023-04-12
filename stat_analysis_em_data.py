import multiGroupTest,dataIO
import matplotlib.pyplot as plt
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
tag= "rei_EM_data"
# Assign the current file path, data subset, and save file name
file = f'./Data/{tag}.csv'
df = pd.read_csv(file)
observation_types = df.obs_type.unique()
df['id'] = df.genotype +'_'+ df.obs_type

# group the dataframe by genotype and sum the columns
binoObj = binominalStats.binominalStats(0,0)
conf_int = list(    )
for i,row in df.iterrows() :
    binoObj.heads = row.observation
    binoObj.total_flips = row.area_microm2
    conf_int.append(binoObj.exact_CI())
df_stat = pd.DataFrame(conf_int)
df_stat = df_stat.rename(columns={"Proportion":"frequency","Lower CI":f"lowCI","Upper CI":f"upCI"})
# calculate the error bar lengths
df_stat['lowCI'] = df_stat['frequency'] - df_stat['lowCI']
df_stat['upCI'] = df_stat['upCI'] - df_stat['frequency']
df_stat = df_stat*1000
df = pd.concat([df,df_stat],axis=1)


for obs_type in observation_types:
    df_temp = df.loc[df.obs_type == obs_type,:]
    f = plt.figure()
    plt.bar(df_temp['genotype'], df_temp['frequency'], yerr=[df_temp['lowCI'], df_temp['upCI'] ], capsize=4)
    plt.gca().set_title(obs_type)
    plt.gca().set_ylabel('observations per 1000 micrometer^2')
    plt.gca().set_xlabel('genotype')
    f.savefig(f'./figures/{tag}_{obs_type}.svg')

#plt.show()
save_file = f'./stats/{tag}_Chi2_FDR_BH.csv'

data_stats = [(row.observation, row.area_microm2 - row.observation) for index, row in df.iterrows()]
# flatten list of tuples to 1-dimensional list
data_stats = [item for sublist in data_stats for item in sublist]

id_stats = df['id'].tolist()
# repeat each value twice
id_stats = [x for x in id_stats for i in range(2)]

set_combinations = [('+/+_breaks','-/-_breaks'),('+/+_severe_split','-/-_severe_split'),('+/+_vesiculation','-/-_vesiculation')]

mtg = multiGroupTest.multiGroupTest(data_stats,id_stats,'Binominal:chi2','all',combination_set =set_combinations)
result = mtg.main()
result.to_csv(save_file)