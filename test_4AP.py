import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiGroupTest
import binominalStats
import numpy as np
from write_pretty_table import write_pretty_table

df = pd.read_csv("./Data/combined_proportion_label.csv")

labels  = df.cluster_label.unique()
labels.sort()

for label in labels:
    stat_df = df.loc[df.cluster_label == label,:]

    save_file = f'./stats/cluster_{str(label).zfill(2)}_Fishers_MedianDiff_FDR_BH.csv'
                
    # Create a multiGroupTest object and set the data, group and test parameters
    mgt = multiGroupTest.multiGroupTest(stat_df.proportion.to_numpy(), stat_df['Group'], 'Fisher:medianDiff', 20000)

    # Run the main method of the multiGroupTest object and save the results
    result_df = mgt.main()
    #result_df.to_csv(save_file, index=False)
    write_pretty_table(result_df,save_file,write=True)