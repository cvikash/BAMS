import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLatentPredictiveLoss(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLatentPredictiveLoss, self).__init__()
        
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