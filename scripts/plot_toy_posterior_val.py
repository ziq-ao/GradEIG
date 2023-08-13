import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

label_list = ["BEEG-\nAP", "UEEG-\nMCMC", "ACE", "PCE", "GradBED"]
marker_list = ["x", "+", "*", "o", "^"]
color_list = ['blue', 'red', 'green', 'purple', 'orange']

def plot_mean_errorbar_columns(tensor):
    # Calculate the mean and standard deviation of each column
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)/np.sqrt(tensor.shape[0])

    # Create x-axis values for each column index
    column_indices = range(tensor.shape[1])

    # Plot the mean and error bar
    for i, color in enumerate(color_list):       
        plt.errorbar(column_indices[i], mean[i], yerr=std[i], fmt='o', 
                     capsize=4,
                     #linewidth=2,
                     #markersize=5,
                     color=color, label=label_list[i])
        
    plt.xticks(column_indices, label_list, fontsize=16)
    plt.ylabel('Posterior entropy', fontsize=18)
    plt.grid(linestyle='--')

def plot_boxplot_columns(tensor):
    # Calculate the mean and standard deviation of each column
    tensor = tensor.numpy()
    #mean = torch.mean(tensor, dim=0)
    #std = torch.std(tensor, dim=0)/np.sqrt(tensor.shape[0])

    # Create x-axis values for each column index
    column_indices = range(tensor.shape[1])

    # Plot the mean and error bar
    plt.boxplot(tensor)    
    
    plt.xticks(column_indices, label_list, fontsize=12)
    plt.ylabel('Posterior entropy', fontsize=14)
    plt.grid(linestyle='--')

noise_std = 0.0001
with open("results/toy/posterior_val_noise_std_{0}.pkl".format(noise_std), "rb") as file:
    mse_stats, entropy_stats = pickle.load(file)
plot_mean_errorbar_columns(entropy_stats)
path = 'figs/toy/posterior_val_{0}'.format(noise_std)+'.eps'
plt.savefig(path, format='eps', bbox_inches='tight', dpi=300)
plt.show()
#plot_boxplot_columns(entropy_stats)
