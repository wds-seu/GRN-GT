import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

cell_types = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

auroc_bce_500 = [
    0.9115,
    0.9158,
    0.9313,
    0.9362,
    0.9021,
    0.9112,
    0.9055
]
auroc_bce_1000 = [
    0.9260,
    0.9332,
    0.9343,
    0.9451,
    0.9205,
    0.9181,
    0.7943
]
auroc_bce = [(x + y) / 2 for x, y in zip(auroc_bce_500, auroc_bce_1000)]

auroc_bbce_500 = [
    0.9139,
    0.9210,
    0.9357,
    0.9374,
    0.9142,
    0.9187,
    0.9028

]
auroc_bbce_1000 = [
    0.9293,
    0.9362,
    0.9412,
    0.9458,
    0.9266,
    0.9268,
    0.8041
]
auroc_bbce = [(x + y) / 2 for x, y in zip(auroc_bbce_500, auroc_bbce_1000)]

# 创建DataFrame
data = pd.DataFrame({
    'Cell Type': cell_types * 2,
    'Condition': ['bce'] * len(cell_types) + ['bbce'] * len(cell_types),
    'AUROC': auroc_bce + auroc_bbce
})

# 创建子图
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# 绘制AUROC柱状图
sns.barplot(x='Cell Type', y='AUROC', hue='Condition', data=data, ax=ax1, palette=['#1f77b4', '#ff7f0e'])
ax1.set_title('Non-specific ChIP-seq', fontsize=12)
ax1.set_ylim(0.8, 1.0)
ax1.set_ylabel('AUROC', fontsize=12)
ax1.set_xlabel('Cell Type', fontsize=12)
ax1.legend()
ax1.set_yticks([i / 10 for i in range(8, 11, 1)])
ax1.set_yticklabels([f'{i / 10:.1f}' for i in range(8, 11, 1)], fontsize=12)

# 调整布局
plt.tight_layout()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图表
# plt.show()
plt.savefig('4_9.pdf', format='pdf')