import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Define the dimensions and parameters
input_dim = ...  # Dimensionality of input vector (N)
hidden_dim = ...  # Hidden layer dimension
num_bins = ...  # Number of bins for histograms
output_dim = input_dim * num_bins  # Output dimension (N * K)
num_bins = ...  # Number of bins for histograms

# Create an instance of the MLPActionPredictor
model = MLPActionPredictor(input_dim, hidden_dim, output_dim, num_bins)
predicted_histograms = model()


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
            predicted_cdf = torch.cumsum(predicted_histograms[i, 0:k], dim=-1)[k]
            target_cdf = torch.cumsum(target_histograms[i, 0:k], dim=-1)[k]
            squared_diff = (predicted_cdf - target_cdf) ** 2
            
  
            loss += squared_diff
    
    return loss
