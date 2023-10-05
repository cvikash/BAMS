import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import helpers_py as pyhelpers

# residual block used in TCN module (see below)
class _ResidualBlock(nn.Module):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dilation_base: int,
        dropout_fn,
        weight_norm: bool,
        nr_blocks_below: int,
        num_layers: int,
        input_size: int,
        target_size: int,
    ):
        super().__init__()

        self.dilation_base = dilation_base  ## dilation base factor
        self.kernel_size = kernel_size     ## kernel size
        self.dropout_fn = dropout_fn      ## dropout function
        self.num_layers = num_layers     ## number of layers
        self.nr_blocks_below = nr_blocks_below ## number of blocks below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.prelu = nn.PReLU() ##  parametric ReLU activation function
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if weight_norm:            ## weight normalization
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step 
        # padding before for making the convolutions causal
        left_padding = (self.dilation_base**self.nr_blocks_below) * (
            self.kernel_size - 1
        )
        x = F.pad(x, (left_padding, 0))
        x = self.dropout_fn(self.prelu(self.conv1(x)))

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = self.prelu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.conv1.in_channels != self.conv2.out_channels:
            residual = self.conv3(residual)
        x = x + residual
        return x
    
## Temporal Convolutional Network (TCN) module with residual blocks and skip connections
class _TCNModule(nn.Module):
    def __init__(
    self, input_size: int, kernel_size: int, num_filters: int, num_layers: int,  dilation_base: int, weight_norm: bool,
    target_size: int,
    dropout: float):

    # Inputs
    # x of shape `(batch_size, input_size, num_timesteps)`

        super().__init__()

        # Defining parameters
        self.input_size = input_size
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(
                num_filters,
                kernel_size,
                dilation_base,
                self.dropout,
                weight_norm,
                i,
                num_layers,
                self.input_size,
                target_size,
            )
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)


    def forward(self, x):
        for res_block in self.res_blocks_list:
            x = res_block(x)

        return x


## MLP for latent predictive loss
class MLPLatentPrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(MLPLatentPrediction, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len

        # Define the layers of the MLP
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim*seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim*seq_len)

        
    def forward(self, zt):
        zt = self.flatten(zt)
        hidden1 = F.relu(self.fc1(zt))
        hidden2 = F.relu(self.fc2(hidden1))
        hidden3 = F.relu(self.fc3(hidden2))
   
        output = self.fc4(hidden3)
        
        output = output.view(output.shape[0], self.output_dim, self.seq_len)
        return output


## MLP for action prediction
class MLPActionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_bins, seq_len):
        super(MLPActionPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len

        # Define the layers of the MLP
        self.flatten = nn.Flatten(start_dim=1, end_dim= -1)
        self.fc1 = nn.Linear(input_dim*seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim*num_bins*seq_len)
        
    def forward(self, zt):
        zt = self.flatten(zt)
        hidden1 = F.relu(self.fc1(zt))
        hidden2 = F.relu(self.fc2(hidden1))
        hidden3 = F.relu(self.fc3(hidden2))
        hidden4 = F.relu(self.fc4(hidden3))
        
        output = self.fc5(hidden4)
        
        # Reshape the output to match the shape of hitograms
        output = output.view(output.shape[0], self.output_dim, self.seq_len, self.num_bins)
        
        # Apply softmax along the bins dimension to get normalized histograms
        h_hat = F.softmax(output, dim=-1)
        
        return h_hat

## MLP for behavior prediction
## MLP for action prediction
class MLPBehaviorPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, predict_window):
        super(MLPBehaviorPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_window = predict_window

        # Define the layers of the MLP
        self.flatten = nn.Flatten(start_dim=1, end_dim= -1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim*predict_window)
        
    def forward(self, zt):
        zt = self.flatten(zt)
        hidden1 = F.relu(self.fc1(zt))
        hidden2 = F.relu(self.fc2(hidden1))
        hidden3 = F.relu(self.fc3(hidden2))
        hidden4 = F.relu(self.fc4(hidden3))
        
        output = self.fc5(hidden4)
        
        # Reshape the output to match the shape of hitograms
        output = output.view(output.shape[0], self.output_dim, self.predict_window)
        
        return output
    
# make histograms from behavior to give targets to action predictor
def make_histograms(observations, num_bins, predict_window):
    ob_hist = torch.empty((observations.shape[0],observations.shape[1], observations.shape[2], num_bins))
    for b in range(observations.shape[0]):
        for k in range(observations.shape[2]):
            for i in range(observations.shape[1]):
                histograms = torch.histc(observations[b,i,k:k+predict_window], bins=num_bins, min=torch.min(observations[b,i,:]), max=torch.max(observations[b,i,:]))
                histograms = histograms/torch.sum(histograms)
                ob_hist[b,i,k,:] = histograms
    return ob_hist


# EMD2 loss function for action predictor
def EMD2_loss(predicted_histograms, target_histograms):
    # Calculate the cumulative distribution function (CDF) for predicted and target histograms
    predicted_cdf = torch.cumsum(predicted_histograms, dim=-1)
    target_cdf = torch.cumsum(target_histograms, dim=-1)
    
    # Calculate the squared difference between the CDFs
    squared_diff = (predicted_cdf - target_cdf) ** 2
    
    # Sum the squared differences for each bin (K) and features (N)
    loss = squared_diff.sum()
    
    return loss

## MSE error for behavior feature prediction
def MSE_loss(predicted_features, target_features):
    loss = torch.nn.MSELoss(reduction='sum')
    return loss(predicted_features, target_features)


def latent_predictive_loss_short(zt,zt_1,window_size):
    loss = 0.0

    # regress short term embeddings to neighboring embeddings in a window
    for b in range(zt.shape[0]):
        for i in range(zt.shape[2]):
            # MLP prediction
            prediction = zt_1[b,:,i]
            # bootstrap target in a small window around the prediction
            if i<window_size:
                idx = np.random.randint(0,i+window_size-1)
                target = zt[b, :,0:i+window_size][:,idx]
            elif i+window_size>=zt.shape[2]:
                idx = np.random.randint(0,zt.shape[2]-i+window_size-1)
                target = zt[b, :,i-window_size:zt.shape[2]-1][:,idx]
            else:
                idx = np.random.randint(0,2*window_size-1)
                target = zt[b, :,i-window_size:i+window_size][:,idx]
            
            ## add the loss for each timepoint
            loss += torch.norm(prediction/torch.norm(prediction) - target/torch.norm(target))**2
    return loss

def latent_predictive_loss_long(zt,zt_1):
    loss = 0.0
    for i in range(zt.shape[0]):
        prediction = zt_1[:,i]
        idx = np.random.randint(0,zt.shape[1]-1)
        target = zt[:,idx]
        loss += torch.norm(prediction/torch.norm(prediction) - target/torch.norm(target))**2
    
    return loss



def custom_lr_scheduler(optimizer, epoch):
    # custom logic for setting the learning rate as a function of the epoch
    if epoch == 100:
        lr = 0.0001
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    


