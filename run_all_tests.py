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

# Iterate over all files in the Data directory
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(data_directory, filename)

        # Read the data from the current file
        df = pd.read_csv(file_path)
        df['id'] = df['species'].str[3] # Assuming genotype and ID are synonymous for identification

        for data_type in ['median_speed_mmPs', 'median_temperature_degC']:
            # Filter the DataFrame to include only the specified columns and drop rows with NaN in the current ratio
            stat_df = df[['id', data_type]].dropna(subset=[data_type])

            # Define save path for the statistical results
            save_file = f'./stats/{filename}_{data_type}_Fishers_MedianDiff_FDR_BH.csv'
            
            # Create a multiGroupTest object and set the data, group and test parameters
            mgt = multiGroupTest.multiGroupTest(stat_df[data_type].to_numpy(), stat_df['id'], 'Fisher:medianDiff', 10000)

            
            # Run the main method of the multiGroupTest object and save the results
            result_df = mgt.main()
            result_df.to_csv(save_file, index=False)
            write_pretty_table(result_df,'')


print("All files processed and results saved.")
