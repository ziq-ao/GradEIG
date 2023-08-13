import torch
from utils.plot import plot_hist_marginals
import torch.multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt

design_dim = 3
noise_std_list = [2,0.5, 0.125]
num_designs = 20
design_list = list(2*torch.rand(num_designs, design_dim)-1) 
for i, noise_std in enumerate(noise_std_list):
    with open("results/gradient_accuracy2/noise_std_{0}.pkl".format(noise_std), "rb") as file:
        gt_results, apnmc_results, vnmc_results, mcmc_results = pickle.load(file)
        gt_eig = [item[0] for item in gt_results]
        gt_grad = [item[1] for item in gt_results]
        gt_grad = torch.cat(gt_grad).view(-1,3)
        apnmc_grad = torch.cat(apnmc_results).view(-1,3)
        vnmc_grad = torch.cat(vnmc_results).view(-1,3)
        mcmc_grad = torch.cat(mcmc_results).view(-1,3)
        x = torch.norm(gt_grad-mcmc_grad,p=2,dim=1)
        y = torch.norm(gt_grad-apnmc_grad,p=2,dim=1)
        z = torch.norm(gt_grad-vnmc_grad,p=2,dim=1)
        max_lim = max(max(x),max(y), max(z))
        plt.figure(figsize=(4, 4))
        plt.plot(x, y, 'ob', label="AP vs MCMC")
        plt.plot(x, z, 'og', label="PCE vs MCMC")
        if i == 0:
            plt.legend(fontsize=16)
        xx = np.linspace(0, max_lim, 100)
        yy = xx
        plt.plot(xx, yy, color="red", linewidth=2)
        plt.axis([0, max_lim, 0, max_lim])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("UEEG-MCMC bias", fontsize=18)
        plt.ylabel("Bias of BEEG-AP/PCE", fontsize=18)
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        mean_eig = np.around((sum(gt_eig)/num_designs).numpy(),2)
        plt.title("Mean of EIGs = "+str(mean_eig), fontsize=18)
        plt.grid(linestyle='--')
        path = 'figs/gradient_accuracy2/'+str(noise_std)+'.eps'
        plt.savefig(path, format='eps', bbox_inches='tight')
        plt.show()
        
        
        plt.figure(figsize=(4, 4))
        plt.plot(y, z, 'o',color='orange', label="PCE vs AP")
        xx = np.linspace(0, max_lim, 100)
        yy = xx
        plt.plot(xx, yy, color="red", linewidth=2)
        plt.axis([0, max_lim, 0, max_lim])
        plt.ticklabel_format(style='sci', scilimits=(0,0))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("BEEG-AP bias", fontsize=18)
        plt.ylabel("PCE bias", fontsize=18)
        mean_eig = np.around((sum(gt_eig)/num_designs).numpy(),2)
        plt.title("Mean of EIGs = "+str(mean_eig), fontsize=18)
        plt.grid(linestyle='--')
        path = 'figs/gradient_accuracy2/'+str(noise_std)+'_pce_ap.eps'
        plt.savefig(path, format='eps', bbox_inches='tight')
        plt.show()
