import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from write_pretty_table import write_pretty_table
import FisherResampling


# Path to the directory containing the data files
file_path = 'data/dataset1_duration_angle_velocity.csv'
# Read the data from the current file
df = pd.read_csv(file_path)

# samples   
n_numbers = df.species.value_counts().to_dict()
# parameters
parameters = ['angle_deg','sacc_dur_sec','speed_deg_per_sec']
p_vals = list()

# Test Durations
for parameter in parameters:
    test = FisherResampling.FisherResamplingTest(df.loc[df.species == "gentoo",parameter],
                                        df.loc[df.species == "rockhopper",parameter],
                                        'medianDiff',20000)
    p_vals.append(test.main())


df = pd.DataFrame({'parameter': parameters, 'p_value': p_vals})
write_pretty_table(df, 'stats/species_comparison_results.txt',True)