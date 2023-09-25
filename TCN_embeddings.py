import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import helpers_py as pyhelpers

# Load and preprocess the behavioral dataset
def preprocess_data(dataset,server):
    # Load the dataset
    behavioral_keys = ['diff_heading', 'speed', 'saccade_rate']
    fish_idx = [1,2,3,5,6,7,8,10,11,12,13,14,15,16,17,18,19]
    data = []
    for i in fish_idx:
        file = "behavior_metrics_final_{}.h5".format(i)
        dataset_path = pyhelpers.locate(dataset,server,file)
        data_fish = pyhelpers.unpackh5(dataset_path)
        # Extract the behavioral data
        behavioral_data = np.array([data_fish[key] for key in behavioral_keys])
        data.append(behavioral_data)
    observations = torch.Tensor(data)  

    return observations



# residual block used in TCN (see below)
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

        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(
            input_dim,
            num_filters,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv1d(
            num_filters,
            output_dim,
            kernel_size,
            dilation=(dilation_base**nr_blocks_below),
        )
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1
            ), nn.utils.weight_norm(self.conv2)

        if input_dim != output_dim:
            self.conv3 = nn.Conv1d(input_dim, output_dim, 1)

    def forward(self, x):
        residual = x

        # first step
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
    # ------
    # x of shape `(batch_size, input_size, num_timesteps)`
    #     Tensor containing the features of the input sequence.

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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLatentPrediction, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        
    def forward(self, zt):
        hidden1 = F.relu(self.fc1(zt))
        hidden2 = F.relu(self.fc2(hidden1))
        hidden3 = F.relu(self.fc3(hidden2))
   
        output = self.fc4(hidden3)
   
        return output


## MLP for action prediction
class MLPActionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_bins):
        super(MLPActionPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.input_dim = input_dim
        
        # Define the layers of the MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, zt):
        hidden1 = F.relu(self.fc1(zt))
        hidden2 = F.relu(self.fc2(hidden1))
        hidden3 = F.relu(self.fc3(hidden2))
        hidden4 = F.relu(self.fc4(hidden3))
        
        output = self.fc5(hidden4)
        
        # Reshape the output to match the shape of hitograms
        output = output.view(-1, self.output_dim, self.num_bins)
        
        # Apply softmax along the bins dimension to get normalized histograms
        h_hat = F.softmax(output, dim=2)
        
        return h_hat

def make_histograms(observations, num_bins):
    ob_hist = torch.empty((observations.shape[0], observations.shape[1], num_bins))
    for k in range(observations.shape[1]):
        for i in range(observations.shape[0]):
            ob_hist[i,k,:] = torch.histc(observations[i,k], bins=num_bins)
    return ob_hist


def EMD2_loss(predicted_histograms, target_histograms):
    loss = 0.0 
    for i in range(predicted_histograms.shape[1]):  # Loop through the features (N)
        for k in range(predicted_histograms.shape[2]):  # Loop through the bins (K)
            # Calculate the cumulative distribution function (CDF) for predicted and target histograms
            predicted_cdf = torch.cumsum(predicted_histograms[i, 0:k], dim=-1)[k]
            target_cdf = torch.cumsum(target_histograms[i, 0:k], dim=-1)[k]
            
            # Calculate the squared difference between the CDFs
            squared_diff = (predicted_cdf - target_cdf) ** 2
            
            # Sum the squared differences for each bin (K)
            loss += squared_diff
    
    return loss

def latent_predictive_loss_short(zt,zt_1,window_size):
    
    loss = 0.0

    # regress short term embeddings to neighboring embeddings in a window
    for i in range(zt.shape[0]):

        # MLP prediction
        prediction = zt_1[:,i]

        # bootstrap target in a small window around the prediction
        if i<window_size:
           idx = np.random.randint(0,i+window_size-1)
           target = zt[:,0:i+window_size][idx]
        elif i+window_size>=zt.shape[1]:
           idx = np.random.randint(0,zt.shape[1]-i+window_size-1)
           target = zt[:,i-window_size:zt.shape[1]-1][idx]
        else:
          idx = np.random.randint(0,2*window_size-1)
          target = zt[:,i-window_size:i+window_size][idx]
        
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


## short term embedding params
input_size = 3
kernel_size = 80
num_filters = 64
num_layers = 4
dilation_base = 2
weight_norm = True
target_size = 32
dropout = 0.05

tcn_model_short = _TCNModule(input_size, kernel_size, num_filters, num_layers,  dilation_base, weight_norm,
    target_size, dropout)

## long term embedding params
input_size = 8
kernel_size = 250
num_filters = 64
num_layers = 5
dilation_base = 4
weight_norm = True
target_size = 32
dropout = 0.05
tcn_model_long = _TCNModule(input_size, kernel_size, num_filters, num_layers,  dilation_base, weight_norm,
    target_size, dropout)

# Load and preprocess data
dataset = "20220708_120112"
server = "roli-1"
observations = preprocess_data(dataset,server)

# Create short-term embeddings using TCNs
short_embeddings = tcn_model_short(observations)  # Assuming observations have shape (num_features, batch_size, num_timesteps


