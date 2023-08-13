from torch import nn, optim
from torch.nn import functional as F
import torch
from utils.plot import plot_hist_marginals
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from torch.utils import data
from nn_ import nets as nn_
from nn_.losses import dsm,sliced_sm

from nsf import distributions as distributions_, flows, transforms
from nn_.nde import MixtureOfGaussiansMADE, MultivariateGaussianMDN

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    input("CUDA not available, do you wish to continue?")
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")
    
class SGD():
    
    def __init__(self, model):
        self.model = model
    
    def train(
            self,
        observations,
        batch_size=100,
        learning_rate=5e-4,
        validation_fraction=0.1,
        stop_after_epochs=20,
        alpha = -1,
        sm_loss_type = "ssm",
    ):

        # Get total number of training examples.
        observations = torch.Tensor(observations)
        num_examples = observations.shape[0]

        # Select random train and validation splits from (parameter, observation) pairs.
        permuted_indices = torch.randperm(num_examples)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        train_indices, val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Dataset is shared for training and validation loaders.
        dataset = data.TensorDataset(
            observations
        )

        # Create train and validation loaders using a subset sampler.
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices),
        )
        val_loader = data.DataLoader(
            dataset,
            batch_size=min(batch_size, num_examples - num_training_examples),
            shuffle=False,
            drop_last=False,
            sampler=SubsetRandomSampler(val_indices),
        )

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Keep track of best_validation log_prob seen so far.
        best_validation_log_prob = -1e100
        # Keep track of number of epochs since last improvement.
        epochs_since_last_improvement = 0
        # Keep track of model with best validation performance.
        best_model_state_dict = None

        epochs = 0
        while True:
            # Train for a single epoch.
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = batch[0].to(device).requires_grad_(True)
                log_prob = self.model.log_prob(inputs)
                if alpha > 0:
                    loss1 = -torch.mean(log_prob)
                    loss2,_,_ = self.score_matching_loss(sm_loss_type)(self.model.log_prob, inputs)
                    loss = loss1+alpha*loss2
                else:
                    loss = -torch.mean(log_prob)
                
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

            epochs += 1

            # Calculate validation performance.
            self.model.eval()
            log_prob_sum = 0
            #with torch.no_grad():
            for batch in val_loader:
                    inputs= batch[0].to(device)
                    log_prob = self.model.log_prob(inputs)
                    if alpha > 0:
                        loss1 = -torch.mean(log_prob)
                        loss2,_,_ = self.score_matching_loss(sm_loss_type)(self.model.log_prob, inputs)
                        loss = loss1+alpha*loss2
                    else:
                        loss = -torch.mean(log_prob)
                    log_prob_sum += -loss.sum().item()
            validation_log_prob = log_prob_sum / num_validation_examples
            
            if epochs % 10 == 0 :
                try:
                    print(epochs, loss1, alpha*loss2)
                except:
                    print(epochs, loss)
                    
            # Check for improvement in validation performance over previous epochs.
            if validation_log_prob > best_validation_log_prob:
                best_validation_log_prob = validation_log_prob
                epochs_since_last_improvement = 0
                best_model_state_dict = deepcopy(self.model.state_dict())
            else:
                epochs_since_last_improvement += 1

            # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self.model.load_state_dict(best_model_state_dict)
                break

    def score_matching_loss(self, loss_type="ssm"):
        if loss_type == "ssm":
            return sliced_sm.sliced_score_matching
        elif loss_type == "ssmvr":
            return sliced_sm.sliced_score_matching_vr
        elif loss_type == "dsm":
            return dsm.dsm
        