import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiGroupTest
import binominalStats
import numpy as np
from write_pretty_table import write_pretty_table

# Path to the directory containing the data files
file_path = 'data/saccade_speeds_combined.csv'


# Read the data from the current file
df = pd.read_csv(file_path)

df['id'] = [f'{b}_{c}' for b,c in list(zip(df.Bodypart.to_list(),df.category.to_list()))]

#sets =[('gcbs', 'gcbi'), ('gchs', 'gchi'), ('gcas', 'gcai'), ('gobs', 'gobi'), ('robs', 'robi')]

# Define save path for the statistical results
save_file = f'./stats/translational_vel_Fishers_MedianDiff_FDR_BH.csv'

# Create a multiGroupTest object and set the data, group and test parameters
mgt = multiGroupTest.multiGroupTest(df['median_translational_vel_mPs'].to_numpy(), df['id'], 'Fisher:medianDiff', 20000,'fdr_bh')

# Run the main method of the multiGroupTest object and save the results
result_df = mgt.main()
result_df.to_csv(save_file, index=False)
write_pretty_table(result_df,'')

# Define save path for the statistical results
save_file = f'./stats/rotational_vel_Fishers_MedianDiff_FDR_BH.csv'

# Create a multiGroupTest object and set the data, group and test parameters
mgt = multiGroupTest.multiGroupTest(df['median_abs_rot_vel_degPs'].to_numpy(), df['id'], 'Fisher:medianDiff', 20000,'fdr_bh')

# Run the main method of the multiGroupTest object and save the results
result_df = mgt.main()
result_df.to_csv(save_file, index=False)
write_pretty_table(result_df,'')


# Define save path for the statistical results
save_file = f'./stats/top_rotational_vel_Fishers_MedianDiff_FDR_BH.csv'

# Create a multiGroupTest object and set the data, group and test parameters
mgt = multiGroupTest.multiGroupTest(df['abs_speed_degPs'].to_numpy(), df['id'], 'Fisher:medianDiff', 20000,'fdr_bh')

# Run the main method of the multiGroupTest object and save the results
result_df = mgt.main()
result_df.to_csv(save_file, index=False)
write_pretty_table(result_df,'')