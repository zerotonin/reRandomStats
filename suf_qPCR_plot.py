import matplotlib.pyplot as plt
import numpy as np

# Data
genotypes = ['+/+', '+/-', '-/-', 'RT +/+', 'RT +/-', 'RT -/-', 'H$_2$O']  # H2O as subscript
brain_values = [1.00, 1.57, 1.64,  0.002,  0.004,  0.005, 0.01]
spine_values = [1.00, 2.68, 2.03, np.nan, np.nan, np.nan, 0.01]

# Plotting
bar_width = 0.35
index = np.arange(len(genotypes))

fig, ax = plt.subplots(figsize=(14, 8))

# Bars
brain_bars = plt.bar(index, brain_values, bar_width, label='Brain', color='navy')
spine_bars = plt.bar(index + bar_width, spine_values, bar_width, label='Spine', color='lightgreen')

# Removed the horizontal line

# Labels, title, custom ticks, and y-axis adjustments (using LaTeX for superscript)
plt.xlabel('Genotype/Control', fontsize=14)
plt.ylabel(r'$2^{-\Delta\Delta C_t}$', fontsize=14)  # 2^-ΔΔCt as superscript
plt.title('Relative Gene Expression (Normalized to WT)', fontsize=16)
plt.xticks(index + bar_width / 2, genotypes, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)

plt.yscale('log')
plt.ylim(0.001, 3)

# Adding value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=10)

add_value_labels(brain_bars)
add_value_labels(spine_bars)

# Show the plot
plt.tight_layout()
plt.grid(axis='y', linestyle='--', linewidth=0.7)

plt.show()
