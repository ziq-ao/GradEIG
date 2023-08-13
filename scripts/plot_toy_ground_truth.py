from bed.apnmc import APNMC, IS_Proposal
from bed.vnmc import VNMC
from simulator.toy2 import Toy
import torch
import numpy as np
import pickle

import matplotlib.pyplot as plt

noise_std = 0.01
with open("results/toy/ground_truth_noise_std_{0}.pkl".format(noise_std), "rb") as file:
    results = pickle.load(file)
results[-1] = results[-2]
#plt.imshow(results[:].transpose(1,0), origin='lower', extent=[0, 1, 0, 1])
plt.figure(figsize=(4,4))
plt.contour(results[:].transpose(1,0), origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(location='top',fraction=0.05, pad=0.05)
plt.grid(linestyle='--', linewidth=2)
# 设置 x 轴和 y 轴刻度标签的字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 添加标签和标题
plt.xlabel('$\lambda_1$', fontsize=18)
plt.ylabel('$\lambda_2$', fontsize=18)
#plt.title('坐标字体调整', fontsize=16)
path = 'figs/toy/ground_truth'+str(noise_std)+'.eps'
plt.savefig(path, format='eps', bbox_inches='tight')
plt.show()
