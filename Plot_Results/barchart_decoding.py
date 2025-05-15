import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

cell_types = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

auroc_dot_500 = [
    0.9764,
    0.9797,
    0.9777,
    0.9652,
    0.9657,
    0.9606,
    0.9353
]
auroc_dot_1000 = [
    0.9810,
    0.9798,
    0.9805,
    0.9732,
    0.9709,
    0.9738,
    0.9422
]
auroc_dot = [(x + y) / 2 for x, y in zip(auroc_dot_500, auroc_dot_1000)]

auroc_concat_500 = [
    0.9842,
    0.9858,
    0.9809,
    0.9847,
    0.9775,
    0.9726,
    0.9483
]
auroc_concat_1000 = [
    0.9854,
    0.9864,
    0.9820,
    0.9871,
    0.9807,
    0.9808,
    0.9512
]
auroc_concat = [(x + y) / 2 for x, y in zip(auroc_concat_500, auroc_concat_1000)]

# 创建DataFrame
data = pd.DataFrame({
    'Cell Type': cell_types * 2,
    'Condition': ['dot'] * len(cell_types) + ['concat'] * len(cell_types),
    'AUROC': auroc_dot + auroc_concat
})

# 创建子图
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# 绘制AUROC柱状图
sns.barplot(x='Cell Type', y='AUROC', hue='Condition', data=data, ax=ax1, palette=['#1f77b4', '#ff7f0e'])
ax1.set_title('Cell-type-specific ChIP-seq', fontsize=12)
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
plt.savefig('4_10.pdf', format='pdf')
