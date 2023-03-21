import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from binominal_stats import exact_CI

#%% Load and clean database
df = pd.read_csv("./Data/student_quiz.csv")
# renaming columns for accesibility
df = df.rename(columns={"Age":"age","Gender":"sex",
                        "1*2*3*4*5*6*7*8*9 R2 // 9*8*7*6*5*4*3*2*1R1":"approximated result",
                        "Which words are more often in the english language?":"k_position"})

# getting the proper groupnames for micorframing
df["microframingID"] = df.Group.copy()
df.microframingID[df.microframingID ==1] = "9->1"
df.microframingID[df.microframingID ==2] = "1->9"

# 
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
ci_dict = exact_CI(k_in_start,total)

f, ax = plt.subplots(figsize=(7, 6))
ax.bar("More words starting with k", ci_dict['Proportion'])
ax.errorbar("More words starting with k", ci_dict['Proportion'], yerr=[[ci_dict['Proportion']-ci_dict['Lower CI']], [ci_dict['Upper CI']-ci_dict['Proportion']]], fmt='none', capsize=5, color='black')
ax.plot([-1,1],[50, 50],'k--')
ax.set_ylabel("Percentage of students believing in K>k")
plt.show()

