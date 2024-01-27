import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

class DataReader(object):
    '''
    Parse Ohio dataset aand read objects
    '''
    def __init__(self, filepath, sampling_interval=5):
        self.filepath = filepath
        self.interval = sampling_interval
        # self.interval_timedelta = timedelta(minutes=sampling_interval)
    
    def read_ohio(self):
        df = pd.read_csv(self.filepath)
        # ids = df['5minute_intervals_timestamp'].to_numpy(dtype=np.float16)
        df.dropna(subset=['cbg'], inplace=True)
        glucose = df['cbg'].to_numpy(dtype=np.float32)
        basal = df['basal'].to_numpy(dtype=np.float32)
        bolus = df['bolus'].to_numpy(dtype=np.float32)
        carbs = df['carbInput'].to_numpy(dtype=np.float32)
        gsr = df['gsr'].to_numpy(dtype=np.float32)

        bolus[np.isnan(bolus)] = 0.0
        basal[np.isnan(basal)] = 0.0
        carbs[np.isnan(carbs)] = 0.0
        gsr[np.isnan(gsr)] = np.nanmean(gsr)

        # output = []
        # for i, row in df.iterrows():
        #     # if i == 0:
        #     #     continue
        #     cbg1 = float(row['cbg'])
        #     # t0 = datetime.fromtimestamp(df['5minute_intervals_timestamp'][i-1])
        #     # t1 = datetime.fromtimestamp(row['5minute_intervals_timestamp'])
        #     # delt = t1 - t0
        #     # if delt <= self.interval_timedelta:
        #     #     output[-1].append(float(cbg1))
        #     # else:
        #     output.append(cbg1)
        data = np.vstack((glucose, basal, bolus, carbs, gsr)).T
        assert not np.isnan(data).any()
        return data
    

class GlucoseDataset(InMemoryDataset):
    def __init__(self, config, client_id, filenames, tag='train', root='', transform=None, pre_transform=None):
        self.client_id = client_id
        self.filenames = filenames
        self.tag = tag
        # print(client_id, filenames)
        self.config = config
        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.n_node, self.mean, self.std_dev = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return self.filenames

    @property
    def processed_file_names(self):
        return [f'{self.client_id}_{self.tag}.pt']

    def load_data(self):
        dataset = None
        for filename in self.filenames:
            data = DataReader(filename, 5).read_ohio()
            if dataset is None:
                dataset = data
            else:
                dataset = np.vstack((dataset,data))
        print('Dataset:', self.tag, 'Size:', len(dataset))
        return dataset
    
    def process(self):
        data = self.load_data()
        # Normalize Data
        mean, std_dev = np.mean(data, axis=0), np.std(data, axis=0)
        data = (data - mean) / std_dev
        
        n_timesteps = data.shape[0]
        n_nodes = data.shape[1]
        sampling_horizon, prediction_horizon = self.config['N_HIST'], self.config['N_PRED']
        window = sampling_horizon + prediction_horizon
        
        # Edge Matrices
        edge_index = torch.ones((2,3*n_nodes), dtype=torch.long)
        edge_attr = torch.ones((3*n_nodes,1))
        
        graphs = []
        shapes = set()
        for i in range(
                n_timesteps - window + 1
            ):
                g = Data()
                # T x N x F
                data_slice = data[i : (i + window)]
                g.edge_index = edge_index
                g.edge_attr = edge_attr
                # (T,N) -> (N,T) [4x12]
                full_window = np.swapaxes(data_slice, 0, 1)
                g.x = torch.FloatTensor(full_window[:, 0:sampling_horizon])
                g.y = torch.FloatTensor(full_window[0, sampling_horizon:])
                graphs.append(g)
                shapes.add(g.x.shape)
                shapes.add(g.y.shape)
                # if i == 0:
                #     print('Graph Shape-X', g.x.shape, 'Graph Shape-Y', g.y.shape)

        # Make the actual dataset
        data, slices = self.collate(graphs)
        torch.save((data, slices, n_nodes, mean, std_dev), self.processed_paths[0])


def split_dataset(dataset: GlucoseDataset, ratio=0.8):
    n_train = int(ratio * len(dataset))
    train = dataset[:n_train]
    val = dataset[n_train:]
    return train, val