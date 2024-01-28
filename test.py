from model import ST_GAT
from torch_geometric.loader import DataLoader
import torch
from data import GlucoseDataset, split_dataset
import copy
from tqdm import tqdm
from utils import average_weights, eval, get_clients_splits
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# Constant config to use througout
parser = argparse.ArgumentParser(description='Arguments for Testing')
parser.add_argument('--plot', type=bool, default=0, help='Whether to plot sample prediction graph')
parser.add_argument('--clients', type=int, default=3, help='Number of clients to simulate')
parser.add_argument('--model', type=str, default='./', help='model file location')
args = parser.parse_args()

config = {
    'BATCH_SIZE': 512,
    'N_PRED': 7,
    'N_HIST': 13,
    'N_NODE': 5,
    'DROPOUT': 0.2,
    'N_CLIENTS': args.clients,
}

def plot_prediction(y_pred, y_truth, config):
    sampling = 30
    # Calculate the truth
    s = y_truth.shape
    y_truth = y_truth.reshape(s[0], config['BATCH_SIZE'], s[-1])
    # just get the first prediction out for the nth node
    y_truth = y_truth[:, :, 0]
    # Flatten to get the predictions for entire test dataset
    y_truth = torch.flatten(y_truth)
    day0_truth = y_truth[:int(1440/sampling)]
    # Calculate the predicted
    s = y_pred.shape
    y_pred = y_pred.reshape(s[0], config['BATCH_SIZE'], s[-1])
    # just get the first prediction out for the nth node
    y_pred = y_pred[:, :, 0]
    # Flatten to get the predictions for entire test dataset
    y_pred = torch.flatten(y_pred)
    # Just grab the first day
    day0_pred = y_pred[:int(1440/sampling)]
    t = [t for t in range(0, 1440, sampling)]
    plt.plot(t, day0_pred, label='ST-GAT')
    plt.plot(t, day0_truth, label='truth', alpha=0.8)
    plt.xlabel('Time (minutes)')
    plt.ylabel('CBG prediction')
    plt.title('Predictions of Blood Glucose over time')
    plt.legend()
    plt.savefig('predicted_times.png')
    print('Figure Saved: predicted_times.png')

def test(config):
    #¬ read in all patients data
    pid_all = [540, 552, 544, 567, 584, 596, 559, 563, 570, 588, 575, 591]
    
    # Simulate Clients
    dict_clients, pid_to_client = get_clients_splits(pid_all, config['N_CLIENTS'])
    train_data, test_data = {}, {}
    print(f'Simulated {config["N_CLIENTS"]} Clients')

    # Load Data
    for client_id, pids in dict_clients.items():
        filenames_test = []
        for pid in pids:
            filenames_test.append(f"data/OhioAll_processed/test/{pid}-ws-testing_processed.csv")
        test_data[client_id] = GlucoseDataset(config, client_id, filenames_test, tag='test')

    global_model = ST_GAT(
                        config = config, 
                        in_channels=config['N_HIST'], 
                        out_channels=config['N_PRED'], 
                        batch_size=config['BATCH_SIZE'], 
                        n_nodes=config['N_NODE'], 
                        heads=8, 
                        dropout=config['DROPOUT']
                        )     
    
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.cuda.current_device())
        device = 'cuda' 
        print('Using CUDA')
        global_model.load_state_dict(torch.load(args.model))
    else:
        device = 'cpu'
        print('Using CPU')
        global_model.load_state_dict(torch.load(args.model,
                                                map_location=torch.device('cpu')))
    global_model.eval()      

    # Set the model to train and send it to device.
    global_model.to(device)

    test_rmse = []
    for client_id in range(config['N_CLIENTS']):
            test_dset = test_data[client_id]
            test_dataloader = DataLoader(test_dset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)
            rmse, y_pred, y_truth = eval(global_model, device, test_dataloader)
            test_rmse.append(rmse)
            print(f'Client {client_id}, Test RMSE: {rmse}')

    print(f'Aggregate Test RMSE: {np.nanmean(test_rmse)}')
    return y_pred, y_truth

if __name__ == '__main__':
     y_pred, y_truth = test(config)
     if args.plot:
          plot_prediction(y_pred, y_truth, config)