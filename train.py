import torch
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import main
import logging
import helpers_py as pyhelpers
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Load and preprocess the behavioral dataset
def preprocess_data(dataset):
    # Load the dataset
    behavioral_keys = ['diff_heading', 'speed', 'saccade_rate', 'saccade_amp', 'saccade_regularity']
    data = []
    for d in dataset:
        file = "behavior_metrics_final_{}.h5".format(d[-1])
        dataset_path = pyhelpers.locate(d[0],d[1],file)
        data_fish = pyhelpers.unpackh5(dataset_path)
        # Extract the behavioral data
        behavioral_data = np.array([data_fish[key][0:172500] for key in behavioral_keys])
        # normalize each behavioral metric
        behavioral_data = ((behavioral_data.T - np.nanmean(behavioral_data, axis = 1))/np.nanstd(behavioral_data, axis=1)).T 
        # replace NaNs with zeros
        nan_indices = np.where(np.isnan(behavioral_data))
        behavioral_data[nan_indices] = 0
        
        # append the data
        data.append(behavioral_data)
    observations = torch.Tensor(data)  

    return observations


# Load and preprocess data

light_cycle_dataset  = [("20220708_120112", "roli-1",  i) for i in range(0, 19) if ((i in [0, 4, 9])==False)] +\
    [("20220715_165749", "roli-1", i, ) for i in range(0, 19) if ((i in [2, 11])==False)]

constant_light_dataset =    [("20220512_170000", "roli-7", i) for i in range(0, 19) if ((i in [2, 3, 5, 7, 11, 16])==False)] +\
    [("20220718_170144", "roli-1", i, ) for i in range(0, 19) if ((i in [6, 10])==False)]

constant_dark_dataset = [("20220601_120000", "roli-7", i) for i in range(0, 19) if ((i in [0, 1, 2, 7, 13, 14, 15, 18])==False)] +\
    [("20220726_170610", "roli-1", i,) for i in range(0, 19) if ((i in [0, 6, 7])==False)]


dataset = light_cycle_dataset + constant_light_dataset + constant_dark_dataset

## load the data
data_whole = preprocess_data(dataset)
num_bins = 30
predict_window = 2*60*3
targets = main.make_histograms(data_whole, num_bins, predict_window)
train_dataloader = DataLoader([[data_whole[i,:,:],targets[i,:,:]]for i in range(data_whole.shape[0])], batch_size=4, shuffle=True)

## setup the logger
current_time = datetime.now().replace(second=0, microsecond=0)

logging.basicConfig(filename= 'training_' + current_time.isoformat() + '.log', level=logging.INFO)
writer = SummaryWriter('logs')

## short term embedding params
input_size = 5
kernel_size = 2*60*2 # fs*60*minutes
num_filters = 16
num_layers = 4
dilation_base = 2
weight_norm = True
target_size = 8
dropout = 0.05

tcn_model_short = main._TCNModule(input_size, kernel_size, num_filters, num_layers,  dilation_base, weight_norm,
    target_size, dropout)

## long term embedding params 
input_size = 5
kernel_size = 2*60*30 # fs*60*minutes
num_filters = 16
num_layers = 5
dilation_base = 4
weight_norm = True
target_size = 8
dropout = 0.05
tcn_model_long = main._TCNModule(input_size, kernel_size, num_filters, num_layers,  dilation_base, weight_norm,
    target_size, dropout)


# training parameters for the networks
num_epochs = 500   # random value at this point
base_learning_rate_tcn = 0.001
weight_decay = 4*1e-4

## TCN model and its training optimizers
optimizer_tcn_short = optim.Adam(tcn_model_short.parameters(), lr=base_learning_rate_tcn, weight_decay=weight_decay)
optimizer_tcn_long = optim.Adam(tcn_model_long.parameters(), lr=base_learning_rate_tcn, weight_decay=weight_decay)

## action predictor and its training optimizer
base_learning_rate_predictor = 0.001*10

input_dim = 16
hidden_dim = 16
output_dim = 5

seq_len = 172500
action_prediction_model = main.MLPActionPredictor(input_dim, hidden_dim, output_dim, num_bins, seq_len)
optimizer_action_predictor = optim.Adam(action_prediction_model.parameters(), lr=base_learning_rate_predictor, weight_decay=weight_decay)

## latent predictor and its training optimizer
latent_loss_hyperparameter = 0.01
window_size_short = 2*60*1  
window_size_long = 2*60*10
input_dim = 8
hidden_dim = 16
output_dim = input_dim



latent_prediction_model_short = main.MLPLatentPrediction(input_dim, hidden_dim, output_dim, seq_len)
latent_prediction_model_long = main.MLPLatentPrediction(input_dim, hidden_dim, output_dim, seq_len)

optimizer_latent_predictor_short = optim.Adam(latent_prediction_model_short.parameters(), lr=base_learning_rate_predictor, weight_decay=weight_decay)
optimizer_latent_predictor_long = optim.Adam(latent_prediction_model_long.parameters(), lr=base_learning_rate_predictor, weight_decay=weight_decay)



for epoch in range(num_epochs):
    running_loss = 0.0
    for (i,batch) in enumerate(train_dataloader):

        # Zero the gradients for TCN optimizers
        optimizer_tcn_short.zero_grad()
        optimizer_tcn_long.zero_grad()
        
        # Zero the gradients for MLP optimizers
        optimizer_action_predictor.zero_grad()
        optimizer_latent_predictor_short.zero_grad()
        optimizer_latent_predictor_long.zero_grad()

        # Forward pass for TCN
        short_embeddings = tcn_model_short(batch)
        long_embeddings = tcn_model_long(batch)

        embeddings = torch.cat((short_embeddings, long_embeddings), 1)

        # Forward pass for action predictor
        predicted_histograms = action_prediction_model(embeddings)

        # Forward pass for latent predictor
        predicted_short_embeddings = latent_prediction_model_short(short_embeddings)
        predicted_long_embeddings = latent_prediction_model_long(long_embeddings)

        # Compute the loss for action predictor
        target_histograms = main.make_histograms(batch, num_bins)
        loss_action_predictor = main.EMD2_loss(predicted_histograms, target_histograms)

        # Compute the loss for latent predictor
        loss_latent_predictor_short = main.latent_predictive_loss_short(short_embeddings, predicted_short_embeddings, window_size_short)
        loss_latent_predictor_long = main.latent_predictive_loss_short(long_embeddings, predicted_long_embeddings, window_size_long)



        # Compute the total loss
        loss = loss_action_predictor + latent_loss_hyperparameter * (loss_latent_predictor_short + loss_latent_predictor_long)
        
        # update the weights
        loss.backward()
        optimizer_tcn_short.step()
        optimizer_tcn_long.step()
        optimizer_action_predictor.step()
        optimizer_latent_predictor_short.step()
        optimizer_latent_predictor_long.step()

        running_loss += loss.item()
        
         # Log loss to the console and to the log file
        logging.info(f'Epoch [{epoch + 1}, Batch [{i + 1}]: Loss: {loss.item()}')

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + i)
        
    # Log average loss for the epoch
    avg_loss = running_loss / len(train_dataloader)
    logging.info(f'Epoch [{epoch + 1}] Average Loss: {avg_loss}')
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

writer.close()
