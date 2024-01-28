import torch
import copy
import numpy as np

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

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def un_z_score(x_normed, mean, std):
    """
    Undo the Z-score calculation
    :param x_normed: torch array, input array to be un-normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    """
    return x_normed * std  + mean

def RMSE(v, v_):
    """
    Mean squared error.
    :param v: torch array, ground truth.
    :param v_: torch array, prediction.
    :return: torch scalar, RMSE averages on all elements of input.
    """
    return torch.sqrt(torch.mean((v_ - v) ** 2))

@torch.no_grad()
def eval(model, device, dataloader, type=''):
    """
    Evaluation function to evaluate model on data
    :param model Model to evaluate
    :param device Device to evaluate on
    :param dataloader Data loader
    :param type Name of evaluation type, e.g. Train/Val/Test
    """
    model.eval()
    model.to(device)

    rmse = 0
    n = 0
    
    # Evaluate model on all data
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch, device).to('cpu')
            truth = batch.y.view(pred.shape).to('cpu')
            if i == 0:
                y_pred = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                y_truth = torch.zeros(len(dataloader), pred.shape[0], pred.shape[1])
                # print(dataloader.dataset.std_dev[0],dataloader.dataset.mean[0])
                # print(pred[0], truth[0])
                # print(pred[0]*dataloader.dataset.std_dev[0]+ dataloader.dataset.mean[0], truth[0]*dataloader.dataset.std_dev[0]+ dataloader.dataset.mean[0])
            truth = un_z_score(truth, dataloader.dataset.mean[0], dataloader.dataset.std_dev[0])
            pred = un_z_score(pred, dataloader.dataset.mean[0], dataloader.dataset.std_dev[0])
            y_pred[i, :pred.shape[0], :] = pred
            y_truth[i, :pred.shape[0], :] = truth
            rmse += RMSE(truth, pred) 
            n += 1
    rmse = rmse / n 
    #get the average score for each metric in each batch
    return rmse, y_pred, y_truth