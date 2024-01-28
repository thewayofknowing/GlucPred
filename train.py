from model import ST_GAT
from torch_geometric.loader import DataLoader
import torch
from data import GlucoseDataset, split_dataset
import copy
from tqdm import tqdm
from utils import average_weights, eval
import numpy as np
import os
import argparse

# Constant config to use througout
parser = argparse.ArgumentParser(description='Arguments for Training')
parser.add_argument('--lr', type=float, default=5e-3, help='adam optimizer learning rate')
parser.add_argument('--decay', type=float, default=0, help='adam optimizer weight decay')
parser.add_argument('--clients', type=int, default=3, help='Number of clients to simulate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (Global)')
parser.add_argument('--local_epochs', type=int, default=10, help='Number of epochs (Local)')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for parallelization')


args = parser.parse_args()

config = {
    'BATCH_SIZE': args.batch_size,
    'EPOCHS': args.epochs,
    'LOCAL_EPOCHS': args.local_epochs,
    'WEIGHT_DECAY': args.decay,
    'INITIAL_LR': args.lr,
    'CHECKPOINT_DIR': './runs',
    'N_PRED': 7,
    'N_HIST': 13,
    'DROPOUT': 0.2,
    # If false, use GCN paper weight matrix, if true, use GAT paper weight matrix
    'USE_GAT_WEIGHTS': True,
    'N_NODE': 5,
    'N_CLIENTS' : args.clients,
    'FRAC_CLIENTS': 1,
    'PRINT_EVERY': 5,
    'SAVE_EVERY': 5,
    'METRICS_FILENAME': f'metrics_{args.lr}_{args.decay}.csv'
}

def local_update_weights(config, model, dataset, device='cpu'):
    # Set mode to train model
    model.train()
    epoch_loss = []

    # Get training, validation loaders
    train_dset, val_dset = split_dataset(dataset)
    train_dataloader = DataLoader(train_dset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dset, batch_size=config['BATCH_SIZE'], shuffle=True)

    # Set optimizer for the local updates
    optimizer = torch.optim.Adam(model.parameters(), lr=config['INITIAL_LR'],
                                        weight_decay=config['WEIGHT_DECAY'])
    loss_fn = torch.nn.MSELoss

    for local_epoch in range(config['LOCAL_EPOCHS']):
        batch_loss = []
        model.train()
        for _, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            y_pred = torch.squeeze(model(batch, device))
            truth = batch.y.view(y_pred.shape)
            # print('Prediction:', y_pred.shape, 'Target', batch.y.shape)
            loss = loss_fn()(y_pred.float(), torch.squeeze(truth).float())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    print('loss', epoch_loss[-1])

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def get_clients_splits(pids, n_clients=4):
    """
    Sample I.I.D. client data from Ohio dataset
    :param num_users:
    :return: dict of client->pid and pid->client
    """
    num_items = int(len(pids)/n_clients)
    dict_users, all_idxs = {}, list(pids)
    pid_to_client = {}
    for i in range(n_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        for pid in dict_users[i]:
            pid_to_client[pid] = i
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users, pid_to_client


def personalized_train_ohio(config):
    #¬ read in all patients data
    pid_all = [540, 552, 544, 567, 584, 596, 559, 563, 570, 588, 575, 591]
    
    # Simulate Clients
    dict_clients, pid_to_client = get_clients_splits(pid_all, config['N_CLIENTS'])
    train_data, test_data = {}, {}

    # Load Data
    for client_id, pids in dict_clients.items():
        filenames_train, filenames_test = [], []
        for pid in pids:
            filenames_train.append(f"data/OhioAll_processed/train/{pid}-ws-training_processed.csv")
            filenames_test.append(f"data/OhioAll_processed/test/{pid}-ws-testing_processed.csv")
        train_data[client_id] = GlucoseDataset(config, client_id, filenames_train, tag='train')
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
    else:
        device = 'cpu'
        print('Using CPU')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_mse = [], []
    val_mse, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    best_val_mse = np.inf

    f = open(config['METRICS_FILENAME'],'w')
    f.close()

    for epoch in tqdm(range(config['EPOCHS'])):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(config['FRAC_CLIENTS'] * config['N_CLIENTS']), 1)
        clients = np.random.choice(range(config['N_CLIENTS']), m, replace=False)

        for client_id in clients:
            print('ClientID', client_id)
            local_model = copy.deepcopy(global_model)
            w, loss = local_update_weights(config, local_model, train_data[client_id], device)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        f = open(config['METRICS_FILENAME'],'a')
        f.write(f'Epoch,{epoch},TRAIN_LOSS,{loss_avg}\n')
        f.close()

        # Calculate avg training accuracy over all users at every epoch
        train_list_mse, val_list_mse = [], []
        global_model.eval()
        for client_id in range(config['N_CLIENTS']):
            train_dset, val_dset = split_dataset(train_data[client_id])
            train_dataloader = DataLoader(train_dset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)
            val_dataloader = DataLoader(val_dset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True)
            train_rmse, _, _ = eval(global_model, device, train_dataloader)
            val_rmse, _, _ = eval(global_model, device, val_dataloader)
            train_list_mse.append(train_rmse)
            val_list_mse.append(val_rmse)
        train_mse.append(sum(train_list_mse)/len(train_list_mse))
        val_mse.append(sum(val_list_mse)/len(val_list_mse))
        f = open(config['METRICS_FILENAME'],'a')
        f.write(f'Epoch,{epoch},TRAIN_RMSE,{train_mse[-1]}\n')
        f.write(f'Epoch,{epoch},VAL_RMSE,{val_mse[-1]}\n')
        f.close()

        # print global training loss after every 'i' rounds
        if (epoch+1) % config['PRINT_EVERY'] == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train RMSE: {:.2f}% \n'.format(train_mse[-1]))
            print('Validation RMSE: {:.2f}% \n'.format(val_mse[-1]))

        if (epoch+1) % config['SAVE_EVERY'] == 0:
            if val_mse[-1] < best_val_mse:
                best_val_mse = val_mse[-1]
                torch.save(global_model.state_dict(), os.path.join(config['CHECKPOINT_DIR'],
                                                        f'model_{config["BATCH_SIZE"]}_{config["INITIAL_LR"]}_{config["WEIGHT_DECAY"]}.pth'))
                print('Model Saved')
                f = open(config['METRICS_FILENAME'],'a')
                f.write(f'Model Saved\n')
                f.close()
    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)


if __name__ == '__main__':
    personalized_train_ohio(config)