"""
Title: Statistical Data Analysis and Visualization of Cognitive Biases
Author: Bart Geurten
Publication: Geurten 2023

This script is for the statistical data analysis and visualization of data collected from quizzes on
cognitive biases. The data is loaded from a CSV file containing student responses to various questions
designed to measure cognitive biases such as base-rate fallacy, availability bias, and framing effect.

The script performs the following steps:
1. Load and clean the data
2. Visualize age and sex distribution of the respondents
3. Analyze and visualize the microframing effect on approximation
4. Analyze and visualize the position of the letter 'k' in words (availability bias)
5. Analyze and visualize the word count for different word forms (ease of recall)
6. Analyze and visualize the base rate fallacy in the brain tumor question
7. Analyze and visualize the framing effect in the framing question
8. Analyze and visualize the base rate fallacy in the 'Steve' question
9. Analyze and visualize the conjunction fallacy in the 'Linda' question
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import binominalStats, multiGroupTest,FisherResampling
import scipy.stats as stats

#%% functuion defs
# ┌──────────────────────┐
# │ ░▒▓█ FUNCTIONS █▓▒░  │
# └──────────────────────┘

def plot_binomial_results(results, xlabels,ylabel,title):
    """
    Plots bar graphs with error bars for a set of binomial statistics.

    Args:
        results (list): A list of dictionaries, where each dictionary contains the results of a binomial analysis.
            Each dictionary should have the following keys: "Proportion", "Lower CI", "Upper CI", and "p_value".
        xlabels (list): A list of strings to use as the x-axis labels for the bar plot.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, res in enumerate(results):
        ax.bar(xlabels[i], res['Proportion'])
        ax.errorbar(xlabels[i], res['Proportion'], yerr=[[res['Proportion']-res['Lower CI']], [res['Upper CI']-res['Proportion']]],
                    fmt='none', capsize=5, color='black')

    ax.plot([-1, len(xlabels)], [50, 50], 'k--')
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_title(title)
    return fig,ax


def analyze_two_choice_question(df, column, count_value):
    """
    Analyzes a single question in a DataFrame by counting the occurrences of each answer option,
    calculating binomial statistics, calculating the exact confidence interval for the proportion,
    and performing a binomial test to determine the p-value.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the data to analyze.
        column (str): The name of the column containing the data for the question to analyze.
        count_value (str): The value to count in the question column.
    
    Returns:
        dict: A dictionary containing the count of each answer option, the binomial statistics,
        the exact confidence interval for the proportion, and the p-value.
    """
    # Count the occurrences of each answer option
    count = df[column].str.count(count_value).value_counts()

    # sometimes the value is not in the dataset, then we have to set the value as 0
    if len(count) == 1:
        count[1] = 0

    # Calculate binomial statistics
    binom = binominalStats.binominalStats(count[1], count.sum())

    # Calculate the exact confidence interval for the proportion
    result = binom.exact_CI()

    # Perform a binomial test to determine the p-value
    p_value = binom.binomial_test()

    # Create a dictionary with the results
    result['count'] = count
    result['p_value'] = p_value

    return result

def write_bino_stats_file(dictionary, filepath):
    """
    Writes a Python dictionary to an ASCII file.
    
    Args:
        dictionary (dict): The dictionary to write to the file.
        filepath (str): The file path to write the dictionary to.
    
    Returns:
        None.
    
    Raises:
        IOError: If the file path is invalid or the file cannot be written.
    """
    # Open the file for writing
    try:
        with open(filepath, 'w') as f:
            # Loop over the dictionary items and write them to the file
            for key, value in dictionary.items():
                f.write(f'{key}: {value}\n')
    except IOError as e:
        raise IOError(f'Error writing dictionary to file: {str(e)}')



#%% loading
# ┌──────────────────────┐
# │ ░▒▓█ LOAD DATA █▓▒░  │
# └──────────────────────┘
df_ai = pd.read_csv("./Data/df_ai.csv")
df_human = pd.read_csv("./Data/df_human.csv")

df_gpt35 = df_ai[df_ai["AI"] == "GPT3_5"]
df_gpt4  = df_ai[df_ai["AI"] == "GPT4"]


#%% Age Sex Overview

# ┌───────────────────────────────────────────────────────────┐
# │ ░▒▓█ VISUALIZE AGE & SEX DISTRIBUTION OF RESPONDENTS █▓▒░ │
# └───────────────────────────────────────────────────────────┘

# Create a new plot with a specified size
f, ax = plt.subplots(figsize=(7, 6))

# Create a distribution plot of the 'age' column with 'sex' as the hue, including a kernel density estimate
sns.displot(data=df_human, x="age", hue="sex", kde=True, ax=ax)

# Display the plot
#plt.show()
#%% Linda

# ┌──────────────────────────────────────────────────┐
# │ ░▒▓█ 'LINDA' QUESTION (CONJUNCTION FALLACY) █▓▒░ │
# └──────────────────────────────────────────────────┘

# Count the occurrences of each answer option for the Linda question
binoH_stats = analyze_two_choice_question(df_human,'Linda',"Linda is a bank teller and is active in the feminist movement.")
bino3_stats = analyze_two_choice_question(df_gpt35,'Linda', 'B')
bino4_stats = analyze_two_choice_question(df_gpt4,'Linda' , 'B')
results =[binoH_stats, bino3_stats,bino4_stats]

# Create a bar plot with error bars to visualize the results
f_linda, ax_linda = plot_binomial_results(results, ['Human ','GPT 3.5','GPT 4'],
                      "Answer indicating Linda is a feminist and bankteller, %",
                      f"p values: {[res['p_value'] for res in results]}")


#statistics for Linda question
data = [binoH_stats['count'][1], binoH_stats['count'][0],binoH_stats['count'][1]+ binoH_stats['count'][0],
        bino3_stats['count'][1], bino3_stats['count'][0],bino3_stats['count'][1]+ bino3_stats['count'][0], 
        bino4_stats['count'][1], bino4_stats['count'][0],bino4_stats['count'][1]+ bino4_stats['count'][0]]
group = ['H','H','H','35','35','35','4','4','4']

mtg = multiGroupTest.multiGroupTest(data,group,"Binominal:chi2",0)
stat_result = mtg.main()
stat_result.to_csv('./stats/Linda_stats.csv')


#%% Microframing Factorial

# ┌───────────────────────────────────────────────────┐
# │ ░▒▓█ ANALYZE & VISUALIZE MICROFRAMING EFFECT █▓▒░ │
# └───────────────────────────────────────────────────┘

# Create a DataFrame with 'factorial framing' and 'microframingID' columns
micro_df = df_human[["factorial framing","microframingID"]]

# Drop rows with missing values
micro_df = micro_df.dropna()

# Keep only rows with numeric values in the 'factorial framing' column
micro_df = micro_df[pd.to_numeric(micro_df['factorial framing'], errors='coerce').notna()]

# Perform Fisher's resampling test on the medians
frs = FisherResampling.FisherResamplingTest(micro_df.loc[micro_df["microframingID"]=="1->9", "factorial framing"],
                                            micro_df.loc[micro_df["microframingID"]=="9->1", "factorial framing"],
                                            "medianDiff",10000)
p_value = frs.main()

# Load a colorblind-friendly palette
palette = sns.color_palette("colorblind")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 6))

# Create a boxplot to visualize the microframing effect
sns.boxplot(x="microframingID", y="factorial framing", data=micro_df,hue="microframingID",
            notch=False, showcaps=False,
            flierprops={"marker": "x"}, dodge=False,
            palette=palette,
            width=.6,ax=ax1)

# Add in points to show each observation
sns.stripplot(x="microframingID", y="factorial framing", data=micro_df,
              size=4, color=".3", linewidth=0)

# Set the title and scale
ax1.set_title(f" Fisher Resampling on medians. p-value: {p_value:.3}")
ax1.set_yscale("log")

# Display the plot
plt.show()




#%% letter k position availability bias / ease of recall
# ┌───────────────────────────────────────────────────┐
# │ ░▒▓█ LETTER 'K' POSITION (AVAILABILITY BIAS) █▓▒░ │
# └───────────────────────────────────────────────────┘

# Get the value counts for the 'k_position' column in the DataFrame
result = df_human.k_position.value_counts()

# Calculate the total count
total  = np.sum(result.values)

# Get the count of responses indicating more words have 'k' at the third position than at the beginning
k_in_start = result["More words have the letter k at the third position, than at the beginning"]

# Perform binomial statistics analysis
binom=binominalStats.binominalStats(k_in_start,total)
bino_stats = binom.exact_CI()
p_value = binom.binomial_test()

# Create a bar plot to visualize the percentage of students believing in K>k
f, ax = plt.subplots(figsize=(7, 6))
ax.bar("More words starting with k", bino_stats['Proportion'])
ax.errorbar("More words starting with k", bino_stats['Proportion'], yerr=[[bino_stats['Proportion']-bino_stats['Lower CI']], [bino_stats['Upper CI']-bino_stats['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1,1],[50, 50],'k--')
ax.set_ylabel("Percentage of students believing in K>k")
ax.set_title(f"p value: {p_value:.3e}")

# Display the plot
plt.show()

#%%  ____n__ vs _____ing ease of recall / availability

# ┌───────────────────────────────────────────────┐
# │ ░▒▓█ WORD COUNT FOR DIFFERENT WORD FORMS █▓▒░ │
# └───────────────────────────────────────────────┘

# Create a DataFrame with only letter-related columns
boxplot_df = df_human[["letter_ing","letter_i","letter_ly","letter_l"]]

# Melt the DataFrame to create a new DataFrame with 'word_count' and 'question' columns
boxplot_df = boxplot_df.melt(var_name='question', value_name='word_count')

# Remove rows with missing values in the 'word_count' column
boxplot_df = boxplot_df.dropna(subset=['word_count'])

# Reset the index
boxplot_df.reset_index(drop=True, inplace=True)

# Load a colorblind-friendly palette
palette = sns.color_palette("colorblind")

# Perform multi-group test using Fisher's mean difference
mtg = multiGroupTest.multiGroupTest(boxplot_df.word_count,boxplot_df.question,"Fisher:meanDiff",10000)
stat_result = mtg.main()

# Save the statistical results to a CSV file
stat_result.to_csv("./Data/letter_stats.csv")

# Create a boxplot to visualize the word count for different word forms
f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=boxplot_df, x="question", y="word_count", hue="question",
            notch=True, showcaps=False,
            flierprops={"marker": "x"}, dodge=False,
            palette=palette)



# Display the plot
plt.show()

#%% tumor question base rate fallacy

# ┌────────────────────────────────────────────────────┐
# │ ░▒▓█ BRAIN TUMOR QUESTION (BASE RATE FALLACY) █▓▒░ │
# └────────────────────────────────────────────────────┘

# The actual probability is 33.2%
tumor_test_prob = 33.2

# Calculate the median of tumor column values
median_value = df_human["tumor"].median()

# Perform a one-sample Wilcoxon signed-rank test comparing tumor values to the actual probability
stat, p_value = stats.wilcoxon(df_human["tumor"] - tumor_test_prob)
print(f'Statistic: {stat}, p-value: {p_value}')

# Create a violin plot of the tumor column values
ax = sns.violinplot(data=df_human, y="tumor", inner="quartile", scale="width")

# Add vertical lines for the median and the actual probability
plt.axhline(y=median_value, color='r', linestyle='--', label=f'Median: {median_value:.2f}')
plt.axhline(y=tumor_test_prob, color='g', linestyle='--', label=f'Statistical probability')

# Add a legend
plt.legend()

# Determine if the median is significantly different from the actual probability
alpha = 0.05
if p_value < alpha:
    ax.set_title(f"Median, and statistical probability sig. different! p-value: {p_value:.3e}")
else:
    ax.set_title(f"Fail to reject the null hypothesis. p-value: {p_value:.3e}")

# Display the plot
plt.show()


##%% framing framing

# ┌─────────────────────────────────────────────┐
# │ ░▒▓█ framing QUESTION (FRAMING EFFECT) █▓▒░ │
# └─────────────────────────────────────────────┘

# Create a DataFrame containing only framing and framingID columns
framing_df = df_human[["framing", "framingID"]]

# Remove rows that do not start with 'P' in the framing column
framing_df = framing_df[framing_df['framing'].str.startswith('P') == True]

# Count occurrences of each combination of framing and framingID
total_n = framing_df.value_counts()
print(total_n)

# Calculate the positive and negative frame counts
pos_frame = np.array([total_n[3], total_n[1]])
neg_frame = np.array([total_n[0], total_n[2]])

# Perform multiple binomial tests between positive and negative framings
mbt = binominalStats.MultipleBinominalTests(pos_frame, neg_frame)
p_value = mbt.perform_test()

# Calculate the exact confidence interval for the positive framing
binom = binominalStats.binominalStats(pos_frame[0], pos_frame.sum())
ci_dict_pos = binom.exact_CI()

# Calculate the exact confidence interval for the negative framing
binom = binominalStats.binominalStats(neg_frame[0], neg_frame.sum())
ci_dict_neg = binom.exact_CI()

# Create a bar plot with error bars to visualize the results
f, ax = plt.subplots(figsize=(7, 6))
ax.bar(["positive", "negative"], [ci_dict_pos['Proportion'], ci_dict_neg['Proportion']])
ax.errorbar("positive", ci_dict_pos['Proportion'], yerr=[[ci_dict_pos['Proportion']-ci_dict_pos['Lower CI']], [ci_dict_pos['Upper CI']-ci_dict_pos['Proportion']]], fmt='none', capsize=5, color='black')
ax.errorbar("negative", ci_dict_neg['Proportion'], yerr=[[ci_dict_neg['Proportion']-ci_dict_neg['Lower CI']], [ci_dict_neg['Upper CI']-ci_dict_neg['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-0.5, 1.5], [50, 50], 'k--')
ax.set_ylabel("Percentage of students chosing to gamble")
ax.set_title(f"p value: {p_value:.3}")

# Display the plot
plt.show()

#%% Steve base rate fallacy

# ┌────────────────────────────────────────────────┐
# │ ░▒▓█ 'STEVE' QUESTION (BASE RATE FALLACY) █▓▒░ │
# └────────────────────────────────────────────────┘

# Count the occurrences of each answer option for the Steve question
steve_count = df_human.Steve.value_counts()

# Calculate the base rate for Steve being a librarian
base_rate = 50000 / 950000

# Calculate binomial statistics for the "Steve is a librarian" option
binom = binominalStats.binominalStats(steve_count[0], steve_count.sum())

# Calculate the exact confidence interval for the proportion of this option
bino_stats = binom.exact_CI()

# Perform a binomial test to determine the p-value
p_value = binom.binomial_test()

# Create a bar plot with error bars to visualize the results
f, ax = plt.subplots(figsize=(7, 6))
ax.bar("Steve is a librarian", bino_stats['Proportion'])
ax.errorbar("Steve is a librarian", bino_stats['Proportion'], yerr=[[bino_stats['Proportion']-bino_stats['Lower CI']], [bino_stats['Upper CI']-bino_stats['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1, 1], [50, 50], 'k--')
ax.set_ylabel("Percentage of students believing that Steve is a librarian")
ax.set_title(f"p value: {p_value:.3e}")

# Display the plot
plt.show()