"""
Title: Statistical Data Analysis and Visualization of Cognitive Biases
Author: Bart Geurten
Publication: Geurten 2023

This script performs statistical data analysis and visualization of data collected from quizzes on cognitive biases. The data is loaded from a CSV file containing student responses to various questions designed to measure cognitive biases such as base-rate fallacy, availability bias, and framing effect.

The script performs the following steps:

Load and clean the data

Functions:

replace_letter_test_questions(df, column_name): Replaces string values in the specified column of a DataFrame with corresponding floating point numbers.
Required Libraries:

pandas

Usage:

Load the CSV files 'student_quiz.csv' and 'ai_quiz.csv' in the 'Data' folder, and place the script in the same directory.
Run the script to perform the statistical data analysis and visualization for the specified cognitive biases.
View the generated plots to analyze the cognitive biases in the student and AI responses.
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

#%%

df_ai.to_csv('./Data/df_ai.csv', index=False)
df_human.to_csv('./Data/df_human.csv', index=False)
