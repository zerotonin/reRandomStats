import multiGroupTest,dataIO
'''
Script:     stat_analysis_OepenEtAl.py
Written by: Bart R.H. Geurten
Date:       15. Jan 2023
Purpose:    Statistical analysis of the results of the Open et al. 2023 article.

This script is used for statistical analysis of the results of the Open et al. 2023 article. 
It uses the `multiGroupTest` class to perform multiple statistical tests on the data, and the 
`dataIO` class to read and manipulate the data. 

The `file_path` variable contains a list of file paths to the csv files containing the original data. 
The `data_subsets` variable contains a list of sets of elements used to select specific subsets of the data.

The script iterates over the file paths, reads the data, selects the desired subset, 
creates a `multiGroupTest` object, runs the statistical tests, and saves the results to a csv file. 

The statistical test used is a Ronald Fisher permutation test to test for differences in median of all 
combinations of two counts. We opted for median differences as the original data are integer counts.
In this case, all possible n out of k combinations of the datasets were calculated. 
All p-values were later corrected with a false discovery rate detection routine from Benjamini and Hochberg.
'''


# file_path contains a list of file paths to csv files containing the original data
file_path = ['./Data/PAL_Development.csv',
             './Data/PAM_Development.csv',
             './Data/PPL1_Development.csv',
             './Data/PPL2_Development.csv',
             './Data/PPM_Development.csv',
             './Data/SEM_SEL_Development.csv']

# data_subsets contains a list of sets of elements that are used to select specific subsets of the data
standard_set = set(['12h', '18h', 'Adult', 'L3'])
prepam_set   = set(['36h', '42h', '48h', 'Adult', 'L3'])
data_subsets = [standard_set,prepam_set,standard_set,standard_set,standard_set,standard_set]

# Iterate over the file paths
for i in range(len(file_path)):
    # Assign the current file path, data subset, and save file name
    file = file_path[i]
    subset = data_subsets[i]
    save_file = file[0:-4]+'_stats_medianDiff_FDR_BH.csv'
    # Create a DataIO object
    dio = dataIO.DataIO()  
    # Read the data from the current file and convert it to a long table format
    id_list,data=dio.wide_table_csv_to_long_table(file)
    # Get the subset of data specified in the data_sub sets list using the get_subset_of_data method
    id_subset,data_subset=dio.get_subset_of_data(id_list,data,subset)
    # Create a multiGroupTest object and set the data, group and test parameters
    mgt = multiGroupTest.multiGroupTest(data_subset,id_subset,'Fisher:medianDiff')
    # Run the main method of the multiGroupTest object
    result_df = mgt.main()
    # Save the result to a csv file
    result_df.to_csv(save_file)
