import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

cell_types = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

auroc_dnabert2_500 = [
    0.9133,
    0.9201,
    0.9332,
    0.9374,
    0.9123,
    0.9214,
    0.8910
]
auroc_dnabert2_1000 = [
    0.9277,
    0.9350,
    0.9423,
    0.9455,
    0.9285,
    0.9224,
    0.8153
]
auroc_dnabert2 = [(x + y) / 2 for x, y in zip(auroc_dnabert2_500, auroc_dnabert2_1000)]

auroc_500 = [
    0.9139,
    0.9210,
    0.9357,
    0.9374,
    0.9142,
    0.9187,
    0.9028
]
auroc_1000 = [
    0.9293,
    0.9362,
    0.9412,
    0.9458,
    0.9266,
    0.9268,
    0.8041
]
auroc = [(x + y) / 2 for x, y in zip(auroc_500, auroc_1000)]

# 创建DataFrame
data = pd.DataFrame({
    'Cell Type': cell_types * 2,
    'Condition': ['DNABERT-2'] * len(cell_types) + ['DeepGene'] * len(cell_types),
    'AUROC': auroc_dnabert2 + auroc
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
plt.savefig('4_11.pdf', format='pdf')
