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

# Load and preprocess the behavioral dataset
def preprocess_data(dataset):
    # Load the dataset
    behavioral_keys = ['diff_heading', 'speed', 'saccade_rate'] #, 'saccade_amp', 'saccade_regularity'
    data = []
    for d in dataset:
        file = "behavior_metrics_final_{}.h5".format(d[-1])
        dataset_path = pyhelpers.locate(d[0],d[1],file)
        data_fish = pyhelpers.unpackh5(dataset_path)
        # Extract the behavioral data
        behavioral_data = np.array([data_fish[key][1500:2000] for key in behavioral_keys])
        # normalize each behavioral metric
        # replace NaNs with zeros
        behavioral_data_standard = ((behavioral_data.T - np.nanmean(behavioral_data, axis = 1))/np.nanstd(behavioral_data, axis=1)).T 

        nan_indices = np.where(np.isnan(behavioral_data_standard))
        behavioral_data_standard[nan_indices] = 0

        # append the data
        data.append(behavioral_data_standard)
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
predict_window = 1

train_dataloader = DataLoader(data_whole[:,1:2,:], batch_size=4, shuffle=False)

## setup the logger
current_time = datetime.now().replace(second=0, microsecond=0)

logging.basicConfig(filename= 'training_mse_version_' + current_time.isoformat() + '.log', level=logging.INFO)
writer = SummaryWriter('logs')

## long term embedding params 
input_size = 1
kernel_size = 5 # fs*60*minutes
num_filters = 3
num_layers = 3
dilation_base = 2
weight_norm = True
target_size = 3
fin_dim = 1
dropout = 0.001
tcn_model_long = main.TCN(input_size, kernel_size, num_filters, num_layers,  dilation_base, weight_norm,
    target_size, dropout, fin_dim, predict_window)


# training parameters for the networks
num_epochs = 200   # random value at this point
base_learning_rate_tcn = 0.001 #0.001
# weight_decay = 4*1e-4

## TCN model and its training optimizers
optimizer_tcn_long = optim.SGD(tcn_model_long.parameters(), lr=base_learning_rate_tcn)


epoch_loss = np.empty(num_epochs, dtype=float)

bad_batch = []
for epoch in range(num_epochs):
    running_loss = 0.0

    print(f'Epoch [{epoch + 1}/{num_epochs}]')

    for (i,batch) in enumerate(train_dataloader):

        batch_loss = 0.0
        # batch = batch.to(device)

        # Forward pass for TCN
        for k in range(predict_window, batch.shape[2]-predict_window-1, predict_window):
            
            # optimizer_beahav_predictor.zero_grad()
            optimizer_tcn_long.zero_grad()

            # Move the batch to the GPU

            long_embeddings = tcn_model_long(batch[:,:, 0:k+1])
            if torch.isnan(long_embeddings).any():
                print("TCN embedding is nan")
                print(tcn_model_long.res_blocks[0].conv1.weight)
                bad_batch.append(batch[:,:, 0:k+1])
            # behav_prediction = behav_prediction_model(long_embeddings[:,k,:])
            # Compute the loss for action predictor
            loss_action_predictor = main.MSE_loss(long_embeddings, batch[:,:,k+1:k+predict_window+1])
             
            # print(loss_action_predictor.item()) 
            batch_loss += loss_action_predictor.item()
            # update the weights
            loss_action_predictor.backward()
            grads = tcn_model_long.fc1.weight.grad
            grad_norm = torch.norm(grads)
            writer.add_scalar('Gradient/conv1', grad_norm, epoch * len(train_dataloader) + i)
            # optimizer_tcn_short.step()
            optimizer_tcn_long.step()
            # optimizer_beahav_predictor.step()

    
        
         
        print(batch_loss)
        # if np.isnan(batch_loss):
        #     break
        running_loss += batch_loss
        # Log loss to the console and to the log file
        logging.info(f'Epoch [{epoch + 1}, Batch [{i + 1}]: Loss: {batch_loss}')

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', batch_loss, epoch * len(train_dataloader) + i)

     
    # Log average loss for the epoch
    avg_loss = running_loss / len(train_dataloader)

    # if np.isnan(avg_loss):
    #     break
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