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
for model_type in ['abm05']:#['abm', 'wfm']:
    data_directory = f'./data/{model_type}/'

    # Iterate over all files in the Data directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            print(f'Processing file: {file_path}')

            # Read the data from the current file
            df = pd.read_csv(file_path)

            if 'replicate' in df.columns:
                df.drop('replicate', axis=1, inplace=True) # Remove replicate column if it exists, as it's not needed for the analysis

            value_col = df.columns.drop('mechanic')[0] 
            df['mechanic'] = df['mechanic'].fillna('no')
            df['id'] = df['mechanic']

            
            # Filter the DataFrame to include only the specified columns and drop rows with NaN in the current ratio
            stat_df = df[['id', value_col]].dropna(subset=[value_col])

            # Define save path for the statistical results
            save_file = f'./stats/{model_type}_{filename[0:-4]}_{value_col}_Fishers_MedianDiff_FDR_BH.csv'
            
            # Create a multiGroupTest object and set the data, group and test parameters
            mgt = multiGroupTest.multiGroupTest(stat_df[value_col].to_numpy(), stat_df['id'], 'Fisher:medianDiff', 10000)

            
            # Run the main method of the multiGroupTest object and save the results
            result_df = mgt.main()
            result_df.to_csv(save_file, index=False)
            write_pretty_table(result_df,'')


print("All files processed and results saved.")
