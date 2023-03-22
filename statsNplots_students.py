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
7. Analyze and visualize the framing effect in the disease question
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

# ┌────────────────────────────────────┐
# │ ░▒▓█ LOAD AND CLEAN THE DATA █▓▒░  │
# └────────────────────────────────────┘


df = pd.read_csv("./Data/student_quiz.csv")
# renaming columns for accesibility
df = df.rename(columns={"Age":"age","Gender":"sex",
                        "1*2*3*4*5*6*7*8*9 R2 // 9*8*7*6*5*4*3*2*1R1":"approximated result",
                        "Which words are more often in the english language?":"k_position",
                        "In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ ing ": "letter_ing",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ n _ ': "letter_n__",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ l y': "letter_ly",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ l _ ': "letter_l_",
                        "Suppose there is an a test for brain tumors, which is 99% specific and sensitive. Meaning there are 1 % false positives. Suppose only 0.5% of the population have brain tumors. How high is the chance that I have a brain tumor if the test was positive?":"tumor",
                        "Imagine that the U.S. is preparing for the outbreak of an unusual disease which is expected to kill 600 people. You have a choice between two programs: ":"disease",
                        'An individual has been described by a neighbour as follows: "Steve is very shy and withdrawn invariably helpful but with little interest in people or in the world of reality. A meek and tidy soul, he has a need for order and structure and a passion for detail" Is Steve more likely to be a librarian or a farmer?':"Steve",
                        'Linda is 31 years old, single, outspoken, and very bright. She majored in philosophy. As a student, she was deeply concerned with issues of discrimination and social justice, and also participated in anti-nuclear demonstrations. Which is more likely?':"Linda"})

# getting the proper groupnames for micorframing
df["microframingID"] = ""
df.loc[df["Group"] == 1, "microframingID"] = "9->1"
df.loc[df["Group"] == 2, "microframingID"] = "1->9"
df["framingID"] = ""
df.loc[df["Group"] == 1, "framingID"] = "positive"
df.loc[df["Group"] == 2, "framingID"] = "negative"


# quantifying the ing and _n_ questions
df = replace_letter_test_questions(df,"letter_ing") 
df = replace_letter_test_questions(df,"letter_n__") 
df = replace_letter_test_questions(df,"letter_ly") 
df = replace_letter_test_questions(df,"letter_l_") 

# bringing the tumor question to percentage
df.tumor = df.tumor * 10




#%% Age Sex Overview

# ┌───────────────────────────────────────────────────────────┐
# │ ░▒▓█ VISUALIZE AGE & SEX DISTRIBUTION OF RESPONDENTS █▓▒░ │
# └───────────────────────────────────────────────────────────┘

f, ax = plt.subplots(figsize=(7, 6))
sns.displot(data=df, x="age", hue="sex", kde=True)

plt.show()
#%% Microframing Factorial

# ┌───────────────────────────────────────────────────┐
# │ ░▒▓█ ANALYZE & VISUALIZE MICROFRAMING EFFECT █▓▒░ │
# └───────────────────────────────────────────────────┘

micro_df = df[["approximated result","microframingID"]]
micro_df = micro_df.dropna()
micro_df = micro_df[pd.to_numeric(micro_df['approximated result'], errors='coerce').notna()]
frs = FisherResampling.FisherResamplingTest(micro_df.loc[micro_df["microframingID"]=="1->9", "approximated result"],
                                            micro_df.loc[micro_df["microframingID"]=="9->1", "approximated result"],
                                            "medianDiff",10000)
p_value = frs.main()

# Load a colorblind-friendly palette
palette = sns.color_palette("colorblind")
ax = sns.boxplot(x="microframingID", y="approximated result", data=micro_df,hue="microframingID",
            notch=False, showcaps=False,
            flierprops={"marker": "x"}, dodge=False,
            palette=palette,
            width=.6)
# Add in points to show each observation
sns.stripplot(x="microframingID", y="approximated result", data=micro_df,
              size=4, color=".3", linewidth=0)

ax.set_title(f" Fisher Resampling on medians. p-value: {p_value:.3}")
ax.set_yscale("log")
plt.show()


#%% letter k position availability bias / ease of recall

# ┌───────────────────────────────────────────────────┐
# │ ░▒▓█ LETTER 'K' POSITION (AVAILABILITY BIAS) █▓▒░ │
# └───────────────────────────────────────────────────┘

result = df.k_position.value_counts()
total  = np.sum(result.values)
k_in_start = result["More words have the letter k at the third position, than at the beginning"]
binom=binominalStats.binominalStats(k_in_start,total)
ci_dict = binom.exact_CI()
p_value = binom.binomial_test()

f, ax = plt.subplots(figsize=(7, 6))
ax.bar("More words starting with k", ci_dict['Proportion'])
ax.errorbar("More words starting with k", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1,1],[50, 50],'k--')
ax.set_ylabel("Percentage of students believing in K>k")
ax.set_title(f"p value: {p_value:.3e}")


plt.show()

#%%  ____n__ vs _____ing ease of recall / availability

# ┌───────────────────────────────────────────────┐
# │ ░▒▓█ WORD COUNT FOR DIFFERENT WORD FORMS █▓▒░ │
# └───────────────────────────────────────────────┘

# make df only with letter questions
boxplot_df = df[["letter_ing","letter_n__","letter_ly","letter_l_"]]
# Melt the DataFrame to create a new DataFrame with 'word_count' and 'question' columns
boxplot_df = boxplot_df.melt(var_name='question', value_name='word_count')
# Remove rows with missing values in the 'word_count' column
boxplot_df = boxplot_df.dropna(subset=['word_count'])
# Reset the index
boxplot_df.reset_index(drop=True, inplace=True)

# Load a colorblind-friendly palette
palette = sns.color_palette("colorblind")

mtg = multiGroupTest.multiGroupTest(boxplot_df.word_count,boxplot_df.question,"Fisher:meanDiff",10000)
stat_result = mtg.main()
stat_result.to_csv("./Data/letter_stats.csv")

f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=boxplot_df, x="question", y="word_count", hue="question",
            notch=True, showcaps=False,
            flierprops={"marker": "x"}, dodge=False,
            palette=palette)
plt.show()
#%% tumor question base rate fallacy

# ┌────────────────────────────────────────────────────┐
# │ ░▒▓█ BRAIN TUMOR QUESTION (BASE RATE FALLACY) █▓▒░ │
# └────────────────────────────────────────────────────┘

# The real propability is 33.2%
tumor_test_prob = 33.2

# Calculate the median
median_value = df["tumor"].median()
# Perform a one-sample Wilcoxon signed-rank test
stat, p_value = stats.wilcoxon(df["tumor"] - tumor_test_prob)
print(f'Statistic: {stat}, p-value: {p_value}')

# Create a violin plot
ax = sns.violinplot(data=df, y="tumor", inner="quartile", scale="width")
# Add vertical lines for the median and the mathematical value
plt.axhline(y=median_value, color='r', linestyle='--', label=f'Median: {median_value:.2f}')
plt.axhline(y=tumor_test_prob, color='g', linestyle='--', label=f'Statistical probability')
# Add a legend
plt.legend()

# Determine if the median is significantly different from the mathematical value
alpha = 0.05
if p_value < alpha:
    ax.set_title(f"Median, and statistical probability sig. different! p-value: {p_value:.3e}")
else:
    ax.set_title("Fail to reject the null hypothesis. p-value: {p_value:.3e}")

# Show the plot
plt.show()

#%% disease framing

# ┌─────────────────────────────────────────────┐
# │ ░▒▓█ DISEASE QUESTION (FRAMING EFFECT) █▓▒░ │
# └─────────────────────────────────────────────┘

# make df only with letter questions
disease_df = df[["disease","framingID"]]
# Drop rows that do not start with 'P' in the response column
disease_df = disease_df[disease_df['disease'].str.startswith('P') == True]
# Count rows that start with 'Programm B' and have 'positive' in the second column
total_n = disease_df.value_counts()
print(total_n)
pos_frame = np.array([total_n[3],total_n[1]])
neg_frame = np.array([total_n[0],total_n[2]])

mbt = binominalStats.MultipleBinominalTests(pos_frame,neg_frame)
p_value = mbt.perform_test()

binom=binominalStats.binominalStats(pos_frame[0],pos_frame.sum())
ci_dict_pos = binom.exact_CI()

binom=binominalStats.binominalStats(neg_frame[0],neg_frame.sum())
ci_dict_neg = binom.exact_CI()

f, ax = plt.subplots(figsize=(7, 6))
ax.bar(["positive", "negative"], [ci_dict_pos['Proportion'], ci_dict_neg['Proportion']])
ax.errorbar("positive", ci_dict_pos['Proportion'], yerr=[[ci_dict_pos['Proportion']-ci_dict_pos['Lower CI']], [ci_dict_pos['Upper CI']-ci_dict_pos['Proportion']]], fmt='none', capsize=5, color='black')
ax.errorbar("negative", ci_dict_neg['Proportion'], yerr=[[ci_dict_neg['Proportion']-ci_dict_neg['Lower CI']], [ci_dict_neg['Upper CI']-ci_dict_neg['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-0.5,1.5],[50, 50],'k--')
ax.set_ylabel("Percentage of students chosing to gamble")
ax.set_title(f"p value: {p_value:.3}")
# Show the plot
plt.show()

#%% Steve base rate fallacy

# ┌────────────────────────────────────────────────┐
# │ ░▒▓█ 'STEVE' QUESTION (BASE RATE FALLACY) █▓▒░ │
# └────────────────────────────────────────────────┘

steve_count=df.Steve.value_counts()
base_rate = 50000/950000
binom=binominalStats.binominalStats(steve_count[0],steve_count.sum())
ci_dict = binom.exact_CI()
p_value = binom.binomial_test()

f, ax = plt.subplots(figsize=(7, 6))
ax.bar("Steve is a farmer", ci_dict['Proportion'])
ax.errorbar("Steve is a farmer", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1,1],[50, 50],'k--')
ax.set_ylabel("Percentage of students believing that Steve is a farmer")
ax.set_title(f"p value: {p_value:.3e}")
plt.show()
#%% Linda

# ┌──────────────────────────────────────────────────┐
# │ ░▒▓█ 'LINDA' QUESTION (CONJUNCTION FALLACY) █▓▒░ │
# └──────────────────────────────────────────────────┘

linda_count=df.Linda.value_counts()
binom=binominalStats.binominalStats(linda_count[0],linda_count.sum())
ci_dict = binom.exact_CI()
p_value = binom.binomial_test()

f, ax = plt.subplots(figsize=(7, 6))
ax.bar("Linda is a feminist and bankteller", ci_dict['Proportion'])
ax.errorbar("Linda is a feminist and bankteller", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1,1],[50, 50],'k--')
ax.set_ylabel("Percentage of students believing that Linda is a feminist and bankteller")
ax.set_title(f"p value: {p_value:.3e}")
plt.show()