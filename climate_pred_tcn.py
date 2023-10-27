import torch
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import main
import logging
import helpers_py as pyhelpers
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.preprocessing import StandardScaler


data_path = '/home/vikash//Notebooks/Behavior/BAMS/DailyDelhiClimateTrain.csv'

df = pd.read_csv(data_path)
df = df.drop(['date'], axis=1)
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)

# Convert the standardized array back to a DataFrame
df_standardized = pd.DataFrame(df_standardized, columns=df.columns)


data_whole = torch.Tensor(df_standardized.values, dtype=torch.float32)

predict_window = 1

train_dataloader = DataLoader(data_whole, batch_size=4, shuffle=True)

## setup the logger
current_time = datetime.now().replace(second=0, microsecond=0)

logging.basicConfig(filename= 'training_mse_version_' + current_time.isoformat() + '.log', level=logging.INFO)
writer = SummaryWriter('logs')

## long term embedding params 
input_size = 3
kernel_size = 2*60 # fs*60*minutes
num_filters = 6
num_layers = 3
dilation_base = 2
weight_norm = True
target_size = 6
fin_dim = 3
dropout = 0.005
tcn_model_long = main.TCN(input_size, kernel_size, num_filters, num_layers,  dilation_base, weight_norm,
    target_size, dropout, fin_dim, predict_window)


# training parameters for the networks
num_epochs = 200   # random value at this point
base_learning_rate_tcn = 0.0005 #0.001
weight_decay = 4*1e-4

## TCN model and its training optimizers
optimizer_tcn_long = optim.Adam(tcn_model_long.parameters(), lr=base_learning_rate_tcn, weight_decay=weight_decay)


## action predictor and its training optimizer
base_learning_rate_predictor = 0.001*10 #0.001

input_dim = 6
hidden_dim = 6
output_dim = 3

behav_prediction_model = main.MLPBehaviorPredictor(input_dim, hidden_dim, output_dim, predict_window)
optimizer_beahav_predictor = optim.Adam(behav_prediction_model.parameters(), lr=base_learning_rate_predictor, weight_decay=weight_decay)

epoch_loss = np.empty(num_epochs, dtype=float)

for epoch in range(num_epochs):
    running_loss = 0.0

    print(f'Epoch [{epoch + 1}/{num_epochs}]')

    for (i,batch) in enumerate(train_dataloader):

        batch_loss = 0.0

        nan_indices = np.where(np.isnan(batch))
        batch[nan_indices] = 0
        # Forward pass for TCN
        for k in range(predict_window, batch.shape[2]-predict_window-1, predict_window):

            optimizer_tcn_long.zero_grad()
            # optimizer_beahav_predictor.zero_grad()
        
            long_embeddings = tcn_model_long(batch[:,:, 0:k+1])
            # behav_prediction = behav_prediction_model(long_embeddings[:,k,:])
            # Compute the loss for action predictor
            loss_action_predictor = main.MSE_loss(long_embeddings, batch[:,:,k+1:k+predict_window+1])
             
            # print(loss_action_predictor.item()) 
            batch_loss += loss_action_predictor.item()
            # update the weights
            loss_action_predictor.backward()
            # optimizer_tcn_short.step()
            optimizer_tcn_long.step()
            # optimizer_beahav_predictor.step()

    
        

        nan_indices = np.where(np.isnan(batch))
        print(nan_indices)
        print(batch_loss)
    
        running_loss += batch_loss
        # Log loss to the console and to the log file
        logging.info(f'Epoch [{epoch + 1}, Batch [{i + 1}]: Loss: {batch_loss}')

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', batch_loss, epoch * len(train_dataloader) + i)
        
    # Log average loss for the epoch
    avg_loss = running_loss / len(train_dataloader)
    epoch_loss[epoch] = avg_loss
    logging.info(f'Epoch [{epoch + 1}] Average Loss: {avg_loss}')
    writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

writer.close()

# ## random snippet to test the training loop
# k = np.random.randint(1000)
# batch = iter(train_dataloader)._next_data()
# batch_loss = torch.tensor(0.0, requires_grad=True)  

# # Zero the gradients for TCN optimizers
# optimizer_tcn_short.zero_grad()
# optimizer_tcn_long.zero_grad()

# # Zero the gradients for MLP optimizers
# optimizer_beahav_predictor.zero_grad()

# # Forward pass for TCN
# # short_embeddings = tcn_model_short(batch)
# long_embeddings = tcn_model_long(batch)

# # embeddings = torch.cat((short_embeddings, long_embeddings), 1)
# embeddings = long_embeddings

# # Forward pass for behavior predictor
# predicted_behav = behav_prediction_model(embeddings[:,:,k])
# # Compute the loss for action predictor
# loss_action_predictor = main.MSE_loss(predicted_behav, batch[:,:,k+1:k+predict_window+1])




# # update the weights
# loss_action_predictor.backward()
# optimizer_tcn_short.step()
# optimizer_tcn_long.step()
# optimizer_beahav_predictor.step()


# print(loss_action_predictor.item())