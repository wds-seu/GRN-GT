import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# 数据准备
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
y = np.array([0.8580, 0.8921, 0.9076, 0.9206, 0.9273])
x_labels = ['10%', '20%', '30%', '40%', '50%']
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# 基准线数据与标签对应关系
baselines = [
    (0.5632, 'MI'),
    (0.5851, 'PCC'),
    (0.7189, 'GNE'),
    (0.8814, 'GENELink'),
    (0.8840, 'GNNLink'),
    (0.8965, 'GRNNLink'),
    (0.9480, 'GRN-GT')
]

plt.figure(figsize=(10, 6))

# 绘制主折线
main_line = plt.plot(x, y, marker='o', color='#7f7f7f', linestyle='-', linewidth=2, markersize=4)


plt.axhline(y=baselines[0][0], color=color_list[0], linestyle='--', linewidth=1.2, alpha=0.8)
plt.text(0.525, baselines[0][0], baselines[0][1], color=color_list[0], va='center', ha='left', fontsize=10)
plt.axhline(y=baselines[1][0], color=color_list[1], linestyle='--', linewidth=1.2, alpha=0.8)
plt.text(0.525, baselines[1][0], baselines[1][1], color=color_list[1], va='center', ha='left', fontsize=10)
plt.axhline(y=baselines[2][0], color=color_list[2], linestyle='--', linewidth=1.2, alpha=0.8)
plt.text(0.525, baselines[2][0], baselines[2][1], color=color_list[2], va='center', ha='left', fontsize=10)
plt.axhline(y=baselines[3][0], color=color_list[3], linestyle='--', linewidth=1.2, alpha=0.8)
plt.text(0.525, baselines[3][0]-0.01, baselines[3][1], color=color_list[3], va='center', ha='left', fontsize=10)
plt.axhline(y=baselines[4][0], color=color_list[4], linestyle='--', linewidth=1.2, alpha=0.8)
plt.text(0.525, baselines[4][0]+0.003, baselines[4][1], color=color_list[4], va='center', ha='left', fontsize=10)
plt.axhline(y=baselines[5][0], color=color_list[5], linestyle='--', linewidth=1.2, alpha=0.8)
plt.text(0.525, baselines[5][0]+0.005, baselines[5][1], color=color_list[5], va='center', ha='left', fontsize=10)
plt.axhline(y=baselines[6][0], color=color_list[6], linestyle='--', linewidth=1.2, alpha=0.8)
plt.text(0.525, baselines[6][0], baselines[6][1], color=color_list[6], va='center', ha='left', fontsize=10)

# 坐标轴设置
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xticks(x, x_labels)
plt.ylim(0.5, 1.0)
plt.ylabel('AUROC', fontsize=12)
plt.xlabel('The amount of supervised training data', fontsize=12)

# 网格和边距调整
# plt.grid(True, alpha=0.3)
plt.subplots_adjust(right=0.85)  # 为右侧标签留出空间

# 显示图表
plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.show()
plt.savefig('4_12.pdf', format='pdf')
