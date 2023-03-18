import multiGroupTest,dataIO
import matplotlib.pyplot as plt
from binominal_stats import exact_CI
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
file_path = ['./Data/cell_culture_data.csv']


# Iterate over the file paths
for i in range(len(file_path)):
    # Assign the current file path, data subset, and save file name
    file = file_path[i]
    save_file = file[0:-4]+'_stats_FishersExactTest_FDR_BH.csv'
    # Create a DataIO object
    dio = dataIO.DataIO()  
    # Read the data from the current file and convert it to a long table format
    id_list,data=dio.wide_table_csv_to_long_table(file)
    # Create a multiGroupTest object and set the data, group and test parameters
    mgt = multiGroupTest.multiGroupTest(data,id_list,'Fisher:exact',0)
    # Run the main method of the multiGroupTest object
    result_df = mgt.main()
    # Save the result to a csv file
    result_df.to_csv(save_file)


