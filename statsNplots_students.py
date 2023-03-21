import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import binominal_stats

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
df = pd.read_csv("./Data/student_quiz.csv")
# renaming columns for accesibility
df = df.rename(columns={"Age":"age","Gender":"sex",
                        "1*2*3*4*5*6*7*8*9 R2 // 9*8*7*6*5*4*3*2*1R1":"approximated result",
                        "Which words are more often in the english language?":"k_position",
                        "In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ ing ": "letter_ing",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ n _ ': "letter_n__",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ l y': "letter_ly",
                        'In four pages of an english novel (about 2000 words), how many words would you expect to find that have the form _ _ _ _ _ l _ ': "letter_l_",})

# getting the proper groupnames for micorframing
df["microframingID"] = ""
df.loc[df["Group"] == 1, "microframingID"] = "9->1"
df.loc[df["Group"] == 2, "microframingID"] = "1->9"

# quantifying the ing and _n_ questions

df = replace_letter_test_questions(df,"letter_ing") 
df = replace_letter_test_questions(df,"letter_n__") 
df = replace_letter_test_questions(df,"letter_ly") 
df = replace_letter_test_questions(df,"letter_l_") 




#%% Age Sex Overview

f, ax = plt.subplots(figsize=(7, 6))
sns.displot(data=df, x="age", hue="sex", kde=True)

plt.show()
#%% Microframing Factorial

f, ax = plt.subplots(figsize=(7, 6))
ax.set_yscale("log")
sns.boxplot(x="microframingID", y="approximated result", data=df,
            width=.6)
# Add in points to show each observation
sns.stripplot(x="microframingID", y="approximated result", data=df,
              size=4, color=".3", linewidth=0)
plt.show()


#%% letter k position availability bias / ease of recall

result = df.k_position.value_counts()
total  = np.sum(result.values)
k_in_start = result["More words have the letter k at the third position, than at the beginning"]
binom=binominal_stats.binominal_stats(k_in_start,total)
ci_dict = binom.exact_CI()
p_value = binom.binomial_test()

f, ax = plt.subplots(figsize=(7, 6))
ax.bar("More words starting with k", ci_dict['Proportion'])
ax.errorbar("More words starting with k", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1,1],[50, 50],'k--')
ax.set_ylabel("Percentage of students believing in K>k")
ax.set_title(f"p value: {p_value}")


plt.show()

#%%  ____n__ vs _____ing

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

sns.boxplot(data=boxplot_df, x="question", y="word_count", hue="question",
            notch=True, showcaps=False,
            flierprops={"marker": "x"}, dodge=False,
            palette=palette)
plt.show()


