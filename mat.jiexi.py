import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 加载 .mat 文件  G:\谷歌浏览器下载\HUnet-main_1\HUnet-main\test_result\DiffPAT_cosine_similarity_2MHz_64_250um\T_retrieved_phase.mat
data = scipy.io.loadmat('G:DiffPAT_cosine_similarity_2MHz_64_250um\T_retrieved_phase.mat')

# 访问 phase_angle 变量
phase_angle = data['phase_angle']

# 查看数组的形状
print(phase_angle.shape)

# 可视化数据（示例：热图）
plt.imshow(phase_angle, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Phase Angle Heatmap')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()


