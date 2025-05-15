import matplotlib.pyplot as plt
import numpy as np

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# Data from the table
bp_values = ['60 bp', '300 bp', '2000 bp', '5000 bp']
DNABERT_3mer = [0.7357, 0.7507, 0.7622]
DNABERT_6mer = [0.7238, 0.7268, 0.7811]
DNABERT_2 = [0.7106, 0.7026, 0.8911, 0.9337]
DeepGene = [0.6985, 0.7162, 0.8975, 0.9351]

# Create a figure with two subplots: one for the bar chart and one for the line graph
# fig1, (ax1) = plt.subplots(1, 1, figsize=(10, 6))
fig2, (ax2) = plt.subplots(1, 1, figsize=(10, 6))


# Bar chart
width = 0.15  # width of the bars
x = np.arange(len(bp_values))

# ax1.bar(x[0:3] - width*1.5, DNABERT_3mer, width, label='DNABERT-3mer', color=color_list[0])
# ax1.bar(x[0:3] - width/2, DNABERT_6mer, width, label='DNABERT-6mer', color=color_list[1])
# ax1.bar(x + width/2, DNABERT_2, width, label='DNABERT-2', color=color_list[2])
# ax1.bar(x + width*1.5, DeepGene, width, label='DeepGene', color=color_list[3])

# Set labels and title for the bar chart
# ax1.set_xlabel('Sequence Length', fontsize=12)
# ax1.set_ylabel('F1-score', fontsize=12)
# ax1.set_xticks(x)
# ax1.set_xticklabels(bp_values)
# ax1.set_ylim(0.60, 1.00)
# ax1.legend()

# Line chart
ax2.plot(bp_values[0:3], DNABERT_3mer, marker='o', label='DNABERT-3mer', color=color_list[0])
ax2.plot(bp_values[0:3], DNABERT_6mer, marker='o', label='DNABERT-6mer', color=color_list[1])
ax2.plot(bp_values, DNABERT_2, marker='o', label='DNABERT-2', color=color_list[2])
ax2.plot(bp_values, DeepGene, marker='o', label='DeepGene', color=color_list[3])

# Set labels and title for the line chart
ax2.set_xlabel('Sequence Length', fontsize=12)
ax2.set_ylabel('F1-score', fontsize=12)
ax2.set_ylim(0.60, 1.00)
ax2.legend(fontsize=12)

# Display the plot

# plt.axhline(y=0.7439, color=color_list[0], linestyle='--', linewidth=1.2, alpha=0.8)
# plt.text(-0.1, 0.7439-0.01, 'DNABERT-6mer(avg):0.7439', color=color_list[0], va='center', ha='left', fontsize=8)
# plt.axhline(y=0.7495, color=color_list[1], linestyle='--', linewidth=1.2, alpha=0.8)
# plt.text(-0.1, 0.7495+0.01, 'DNABERT-3mer(avg):0.7495', color=color_list[1], va='center', ha='left', fontsize=8)
# plt.axhline(y=0.8095, color=color_list[2], linestyle='--', linewidth=1.2, alpha=0.8)
# plt.text(-0.1, 0.8095-0.01, 'DNABERT-2(avg):0.8095', color=color_list[2], va='center', ha='left', fontsize=8)
# plt.axhline(y=0.8118, color=color_list[3], linestyle='--', linewidth=1.2, alpha=0.8)
# plt.text(-0.1, 0.8118+0.01, 'DeepGene(avg):0.8118', color=color_list[3], va='center', ha='left', fontsize=8)


plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()


plt.savefig('3_8.pdf', format='pdf')