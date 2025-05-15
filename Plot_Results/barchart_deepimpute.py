import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

cell_types = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

auroc_unimputed_500 = [
    0.9633,
    0.9720,
    0.9748,
    0.9764,
    0.9468,
    0.9418,
    0.8152
]
auroc_unimputed_1000 = [
    0.9622,
    0.9658,
    0.9728,
    0.9782,
    0.9455,
    0.9321,
    0.8414
]
auroc_unimputed = [(x + y)/2 for x, y in zip(auroc_unimputed_500, auroc_unimputed_1000)]

auroc_imputed_500 = [
    0.9632,
    0.9701,
    0.9747,
    0.9762,
    0.9455,
    0.9316,
    0.8307
]
auroc_imputed_1000 = [
    0.9620,
    0.9647,
    0.9729,
    0.9777,
    0.9498,
    0.9319,
    0.8378
]
auroc_imputed = [(x + y)/2 for x, y in zip(auroc_imputed_500, auroc_imputed_1000)]

# 创建DataFrame
data = pd.DataFrame({
    'Cell Type': cell_types * 2,
    'Condition': ['unimputed'] * len(cell_types) + ['imputed'] * len(cell_types),
    'AUROC': auroc_unimputed + auroc_imputed
})

# 创建子图
fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

# 绘制AUROC柱状图
sns.barplot(x='Cell Type', y='AUROC', hue='Condition', data=data, ax=ax1, palette=['#1f77b4', '#ff7f0e'])
ax1.set_title('STRING', fontsize=12)
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
plt.savefig('4_8.pdf', format='pdf')