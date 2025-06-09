import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiGroupTest
import binominalStats
import numpy as np
from write_pretty_table import write_pretty_table

# Path to the directory containing the data files
file_path = 'data/comb_durations.csv'


# Read the data from the current file
df = pd.read_csv(file_path)

df['id'] = [f'{s[0]}{d[0]}{b[0]}{t[0]}' for s,d,b,t in list(zip(df.species.to_list(),df.dataset.to_list(),df.bodypart.to_list(),df.saccade_type.to_list()))]

sets =[('gcbs', 'gcbi'), ('gchs', 'gchi'), ('gcas', 'gcai'), ('gobs', 'gobi'), ('robs', 'robi'), ("gobs","robs")]

# Define save path for the statistical results
save_file = f'./stats/durations_Fishers_MedianDiff_FDR_BH.csv'

# Create a multiGroupTest object and set the data, group and test parameters
mgt = multiGroupTest.multiGroupTest(df['saccade_duration_msec'].to_numpy(), df['id'], 'Fisher:meanDiff', 20000,'fdr_bh',sets)

# Run the main method of the multiGroupTest object and save the results
result_df = mgt.main()
result_df.to_csv(save_file, index=False)
write_pretty_table(result_df,'')