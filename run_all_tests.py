import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiGroupTest
import binominalStats
import numpy as np
from write_pretty_table import write_pretty_table
'''
Script: stat_analysis_Bergers_2024.py
Written by: Bart R.H. Geurten
Date: 28. Apr 2024
Purpose: Statistical analysis of activity and speed ratios of Drosophila mutants for Miriam Bergers' 2024 Master Thesis.

This script is developed for analyzing the effects of environmental light changes on different Drosophila mutants, 
forming a part of Miriam Bergers' Master Thesis in 2024. The analysis investigates the dependence of activity and 
speed ratios on light variations, applying robust statistical methodologies to determine significant differences 
across mutants.

The multiGroupTest class is used to perform multiple statistical tests, evaluating the differences in activity and 
speed ratios among various genotypes under altered light conditions. Results from these analyses are intended to 
help understand how Drosophila mutants react to environmental changes, providing insights into their adaptive behaviors.

Data for this analysis is read from a series of CSV files located in the 'Data' directory, each representing different 
experimental conditions or mutant groups. Results are compiled and saved for further interpretation and discussion 
in the thesis.
'''

# Path to the directory containing the data files
data_directory = './data'
fruc_pairs =   [('female_18.15', 'male_18.15'), 
                ('male_26.2', 'female_26.2'),
                ('male_26.6', 'female_26.6'),
                ('female_5.6', 'male_5.6'),
                ('female_15.1', 'male_15.1'),
                ('female_10.6', 'male_10.6'),
                ('female_26.3', 'male_26.3'),
                ('female_22.9', 'male_22.9'),
                ('female_19.2', 'male_19.2'),
                ('male_7.6', 'female_7.6'), 
                ('male_14.0', 'female_14.0'), 
                ('male_8.0', 'female_8.0'),
                ('male_2.0', 'female_2.0')]
# Ensure the stats directory exists
salt_pairs =   [('male_1.0', 'female_1.0'), 
                ('male_18.8', 'female_18.8'), 
                ('female_6.0', 'male_6.0'), 
                ('female_15.0', 'male_15.0'), 
                ('female_12.0', 'male_12.0'), 
                ('male_4.0', 'female_4.0'), 
                ('male_8.0', 'female_8.0')]

combis =[fruc_pairs, salt_pairs]

c =0
# Iterate over all files in the Data directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_directory, filename)

        # Read the data from the current file
        df = pd.read_csv(file_path)
        df['id']= df.stimulus_01_name+"_"+ df.stimulus_01_amplitude.astype(str)
 
        for ratio in ['preference_index', 'decision_duration_index']:
            # Filter the DataFrame to include only the specified columns and drop rows with NaN in the current ratio
            stat_df = df[['id', ratio]].dropna(subset=[ratio])

            # Define save path for the statistical results
            save_file = f'./stats/{filename}_{ratio}_Fishers_MedianDiff_FDR_BH_pooled.csv'
            
            # Create a multiGroupTest object and set the data, group and test parameters
            mgt = multiGroupTest.multiGroupTest(stat_df[ratio].to_numpy(), stat_df['id'], 'Fisher:medianDiff', 20000,
                                                  correction_type='fdr_bh')#,combination_set=combis[c])
        

        
            # Run the main method of the multiGroupTest object and save the results
            result_df = mgt.main()
            result_df.to_csv(save_file, index=False)
            write_pretty_table(result_df,save_file.replace('.csv', '.txt'))
        c += 1


print("All files processed and results saved.")
