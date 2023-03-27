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

def replace_letter_test_questions(df, column_name):
    """
    Replace string values in the specified column of a DataFrame with corresponding floating point numbers.
    
    The input DataFrame is the result of a student questionnaire that 234 students filled out. The function
    assumes that the string values in the column are specific predefined ranges and replaces them with
    their midpoints as floating point numbers.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the student questionnaire data.
        column_name (str): The name of the column to replace string values with their corresponding floating point numbers.
    
    Returns:
        pandas.DataFrame: The updated DataFrame with replaced values in the specified column.
    """
    # Define a dictionary to map the string values to their corresponding floating point numbers
    answer_translator = {"0": 0.0, "1-2": 1.5, "3-4": 3.5, "5-7": 6, "8-10": 9, "11-15": 13, "16+": 21}

    # Iterate through the key-value pairs in the answer_translator dictionary
    for key, value in answer_translator.items():
        # Replace the string values in the specified column with their corresponding floating point numbers
        df.loc[df[column_name] == key, column_name] = value

    return df


#%% Load and clean database

# ┌────────────────────────────────────────┐
# │ ░▒▓█ LOAD AND CLEAN STUDENT DATA █▓▒░  │
# └────────────────────────────────────────┘


df_human = pd.read_csv("./Data/student_quiz.csv")
# renaming columns for accesibility
df_human = df_human.rename(columns={"Age":"age","Gender":"sex",
                        "1*2*3*4*5*6*7*8*9 R2 // 9*8*7*6*5*4*3*2*1R1":"factorial framing",
                        "Which words are more often in the english language?":"k_position",
                        "In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ ing ": "letter_ing",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ n _ ': "letter_i",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ l y': "letter_ly",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ l _ ': "letter_l",
                        "Suppose there is an a test for brain tumors, which is 99% specific and sensitive. Meaning there are 1 % false positives. Suppose only 0.5% of the population have brain tumors. How high is the chance that I have a brain tumor if the test was positive?":"tumor",
                        "Imagine that the U.S. is preparing for the outbreak of an unusual disease which is expected to kill 600 people. You have a choice between two programs: ":"framing",
                        'An individual has been described by a neighbour as follows: "Steve is very shy and withdrawn invariably helpful but with little interest in people or in the world of reality. A meek and tidy soul, he has a need for order and structure and a passion for detail" Is Steve more likely to be a librarian or a farmer?':"Steve",
                        'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. Which is more likely?':"Linda"})

# getting the proper groupnames for micorframing
df_human["microframingID"] = ""
df_human.loc[df_human["Group"] == 1, "microframingID"] = "9->1"
df_human.loc[df_human["Group"] == 2, "microframingID"] = "1->9"
df_human["framingID"] = ""
df_human.loc[df_human["Group"] == 1, "framingID"] = "positive"
df_human.loc[df_human["Group"] == 2, "framingID"] = "negative"


# quantifying the ing and _n_ questions
df_human = replace_letter_test_questions(df_human,"letter_ing") 
df_human = replace_letter_test_questions(df_human,"letter_i") 
df_human = replace_letter_test_questions(df_human,"letter_ly") 
df_human = replace_letter_test_questions(df_human,"letter_l") 

# bringing the tumor question to percentage
df_human.tumor = df_human.tumor * 10


# ┌───────────────────────────────────┐
# │ ░▒▓█ LOAD AND CLEAN AI DATA █▓▒░  │
# └───────────────────────────────────┘
df_ai = pd.read_csv("./Data/ai_quiz.csv")
df_ai = df_ai.rename(columns={"care_bear_neg":"framing_neg","care_bear_pos":"framing_pos"})

#
#%% Age Sex Overview

# ┌───────────────────────────────────────────────────────────┐
# │ ░▒▓█ VISUALIZE AGE & SEX DISTRIBUTION OF RESPONDENTS █▓▒░ │
# └───────────────────────────────────────────────────────────┘

# Create a new plot with a specified size
f, ax = plt.subplots(figsize=(7, 6))

# Create a distribution plot of the 'age' column with 'sex' as the hue, including a kernel density estimate
sns.displot(data=df_human, x="age", hue="sex", kde=True)

# Display the plot
plt.show()

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

# Create a boxplot to visualize the microframing effect
ax = sns.boxplot(x="microframingID", y="factorial framing", data=micro_df,hue="microframingID",
            notch=False, showcaps=False,
            flierprops={"marker": "x"}, dodge=False,
            palette=palette,
            width=.6)

# Add in points to show each observation
sns.stripplot(x="microframingID", y="factorial framing", data=micro_df,
              size=4, color=".3", linewidth=0)

# Set the title and scale
ax.set_title(f" Fisher Resampling on medians. p-value: {p_value:.3}")
ax.set_yscale("log")

# Display the plot
plt.show()
# Get the value counts for the 'k_position' column in the DataFrame
result = df_human.k_position.value_counts()

# Calculate the total count
total  = np.sum(result.values)

# Get the count of responses indicating more words have 'k' at the third position than at the beginning
k_in_start = result["More words have the letter k at the third position, than at the beginning"]

# Perform binomial statistics analysis
binom=binominalStats.binominalStats(k_in_start,total)
ci_dict = binom.exact_CI()
p_value = binom.binomial_test()

# Create a bar plot to visualize the percentage of students believing in K>k
f, ax = plt.subplots(figsize=(7, 6))
ax.bar("More words starting with k", ci_dict['Proportion'])
ax.errorbar("More words starting with k", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1,1],[50, 50],'k--')
ax.set_ylabel("Percentage of students believing in K>k")
ax.set_title(f"p value: {p_value:.3e}")

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
ci_dict = binom.exact_CI()
p_value = binom.binomial_test()

# Create a bar plot to visualize the percentage of students believing in K>k
f, ax = plt.subplots(figsize=(7, 6))
ax.bar("More words starting with k", ci_dict['Proportion'])
ax.errorbar("More words starting with k", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
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
ci_dict = binom.exact_CI()

# Perform a binomial test to determine the p-value
p_value = binom.binomial_test()

# Create a bar plot with error bars to visualize the results
f, ax = plt.subplots(figsize=(7, 6))
ax.bar("Steve is a librarian", ci_dict['Proportion'])
ax.errorbar("Steve is a librarian", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1, 1], [50, 50], 'k--')
ax.set_ylabel("Percentage of students believing that Steve is a librarian")
ax.set_title(f"p value: {p_value:.3e}")

# Display the plot
plt.show()

#%% Linda

# ┌──────────────────────────────────────────────────┐
# │ ░▒▓█ 'LINDA' QUESTION (CONJUNCTION FALLACY) █▓▒░ │
# └──────────────────────────────────────────────────┘

# Count the occurrences of each answer option for the Linda question
linda_count = df_human.Linda.value_counts()


# Calculate binomial statistics for the "Linda is a feminist and bankteller" option
binom = binominalStats.binominalStats(linda_count[0], linda_count.sum())

# Calculate the exact confidence interval for the proportion of this option
ci_dict = binom.exact_CI()

# Perform a binomial test to determine the p-value
p_value = binom.binomial_test()

# Create a bar plot with error bars to visualize the results
f, ax = plt.subplots(figsize=(7, 6))
ax.bar("Linda is a feminist and bankteller", ci_dict['Proportion'])
ax.errorbar("Linda is a feminist and bankteller", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1, 1], [50, 50], 'k--')
ax.set_ylabel("Percentage of students believing that Linda is a feminist and bankteller")
ax.set_title(f"p value: {p_value:.3e}")

# Display the plot
plt.show()