import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt

label_list = ["BEEG-AP", "UEEG-MCMC", "ACE", "PCE", "GradBED"]
marker_list = ["x", "+", "*", "o", "^"]
#marker_list = [".", ".", ".", ".", "."]
color_list = ['blue', 'red', 'green', 'purple', 'orange']


noise_std = 0.01
with open("results/toy/noise_std_{0}.pkl".format(noise_std), "rb") as file:
    results = pickle.load(file)
    
    plt.figure(figsize=(4,4))
    for i, single_results in enumerate(results):
        for j, single_points in enumerate(single_results):
            plt.plot(
                single_points[0],
                single_points[1],
                alpha=0.5,
                marker=marker_list[i],
                color=color_list[i],
                zorder=j,  # Set the z-order based on the index
                label = label_list[i] if j==0 else None
            )
    
    plt.axis([0, 1, 0, 1.05])
    # Create a new legend with only markers
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, handlelength=0, handletextpad=1,fontsize=14)
    plt.grid(linestyle='--')
    # 设置 x 轴和 y 轴刻度标签的字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # 添加标签和标题
    plt.xlabel('$\lambda_1$', fontsize=16)
    plt.ylabel('$\lambda_2$', fontsize=16)
    #plt.title('坐标字体调整', fontsize=16)
    path = 'figs/toy/'+str(noise_std)+'.eps'
    plt.savefig(path, format='eps', bbox_inches='tight')
    plt.show()


noise_std = 0.0001
with open("results/toy/noise_std_{0}.pkl".format(noise_std), "rb") as file:
    results = pickle.load(file)
    
    plt.figure(figsize=(4,4))
    for i, single_results in enumerate(results):
        for j, single_points in enumerate(single_results):
            plt.plot(
                single_points[0],
                single_points[1],
                alpha=0.5,
                marker=marker_list[i],
                color=color_list[i],
                zorder=j,  # Set the z-order based on the index
                label = label_list[i] if j==0 else None
            )
    
    plt.axis([0, 1, 0, 1.05])
    # Create a new legend with only markers
    handles, labels = plt.gca().get_legend_handles_labels()
    #plt.legend(handles=handles, labels=labels, handlelength=0, handletextpad=1,fontsize=12)
    plt.grid(linestyle='--')
    # 设置 x 轴和 y 轴刻度标签的字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 添加标签和标题
    plt.xlabel('$\lambda_1$', fontsize=16)
    plt.ylabel('$\lambda_2$', fontsize=16)
    #plt.title('坐标字体调整', fontsize=16)
    path = 'figs/toy/'+str(noise_std)+'.eps'
    plt.savefig(path, format='eps', bbox_inches='tight')
    plt.show()