import os
os.environ['DGLBACKEND'] = 'pytorch'
import sys
import torch
import math
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix, linalg
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch.nn.functional as F
from tqdm import tqdm
import time

import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info, split_dataset, Subset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import random_split

class MiniShuffleDataset(DGLDataset):
    '''
    This class import the whole dataset.
    '''
    def __init__(self):
        save_dir = '/home/vincent/sat/sat_selection_light/data/mini_shuffle'
        self.graph_path = os.path.join(save_dir, 'mini_shuffle.bin')
        self.info_path = os.path.join(save_dir, 'mini_shuffle_info.pkl')
        super().__init__(name='mini_shuffle', save_dir=save_dir)
    
    # def process(self):
    #     return
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i].float(), self.graph_info[i]
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        return

    def load(self):
        # load processed data from directory `self.save_path`
        print("Loading data...")
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.labels = label_dict['label']
        # self.graph_info = load_info(self.info_path)
        self.graph_info = [-1] * len(self.graphs)   # Comment out graph_info as loading is too slow
        # if self.idx_list is None:
        #     self.graphs, label_dict = load_graphs(self.graph_path)
        #     self.labels = label_dict['label']
        #     # self.graph_info = load_info(self.info_path)
        #     self.graph_info = [-1] * len(self.graphs)   # Comment out graph_info as loading is too slow
        # else:
        #     idx_list = list(map(int, self.idx_list))   # For some reason have to manually map the index to int otherwise load_graphs doesn't work
        #     # graph_info = pkl.load(open(self.info_path, 'rb'))
        #     # self.graph_info = graph_info[idx_list]
        #     self.graphs, label_dict = load_graphs(self.graph_path, idx_list=idx_list)
        #     self.labels = label_dict['label'][idx_list]
        #     self.graph_info = [-1] * len(idx_list)
        print(f"Dataset has been loaded. Length: {len(self.graphs)}")

    def has_cache(self):
        if not os.path.exists(self.graph_path):
            raise ValueError(f"The dataset cannot be found at: {self.graph_path}")
        return True 

class MiniShuffleDataModule():
    def __init__(self, sat_graph_type = 'hetero', add_clause_pe = False, batch_size: int = 32, num_workers = 32, debug = False, split_idx = 0) :
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.split_idx = split_idx
        self.graph_type = sat_graph_type
        self.add_clause_pe = add_clause_pe
    
    def setup(self, stage: str):
        # Use pre-generated data split
        train_idx_path = '/home/vincent/sat/sat_selection_light/data/mini_shuffle/mini_shuffle_train_idx.npy'
        test_idx_path = '/home/vincent/sat/sat_selection_light/data/mini_shuffle/mini_shuffle_test_idx.npy'

        all_train_idx = np.load(train_idx_path)
        all_test_idx = np.load(test_idx_path)

        train_idx = all_train_idx[self.split_idx]
        test_idx = all_test_idx[self.split_idx]
        
        all_data = MiniShuffleDataset()

        train_data = Subset(all_data, list(train_idx)) 
        self.train_data, self.val_data = random_split(train_data, [0.875, 0.125])
        self.test_data = Subset(all_data, list(test_idx)) 
    
    # def _get_node_type(self, data):
    #     # Same node type used by Wenyi. [pos_literal, neg_literal, clause]
    #     if not data.graphs[0].is_homogeneous:  # Heterogeneous graph
    #         for i in range(len(data)):
    #             hg = data.graphs[i]
    #             num_lit = hg.ndata['_ID']['pos_lit'].shape[0]
    #             num_clause = hg.ndata['_ID']['clause'].shape[0]
    #             data.graphs[i].nodes['pos_lit'].data['node_type'] = F.one_hot(torch.zeros((num_lit), dtype=torch.long), num_classes=3)
    #             data.graphs[i].nodes['neg_lit'].data['node_type'] = F.one_hot(torch.ones((num_lit), dtype=torch.long), num_classes=3)
    #             data.graphs[i].nodes['clause'].data['node_type'] = F.one_hot(2*torch.ones((num_clause), dtype=torch.long), num_classes=3)
 
    #             # data.graphs[i].ndata['onehot']['pos_lit'] = F.one_hot(torch.zeros((num_lit), dtype=torch.long), num_classes=3)
    #             # data.graphs[i].ndata['onehot']['neg_lit'] = F.one_hot(torch.ones((num_lit), dtype=torch.long), num_classes=3)
    #             # data.graphs[i].ndata['onehot']['clause'] = F.one_hot(2*torch.ones((num_clause), dtype=torch.long), num_classes=3)
    #     else:
    #         for i in range(len(data)):
    #             # num_var = data.graph_info[i]['num_var']
    #             # num_clause = data.graph_info[i]['num_clause']
    #             num_var = data.labels[i][0]
    #             num_clause = data.labels[i][1]
    #             num_nodes = num_var * 2 + num_clause + 1  # The node_0 is dumb node existed during graph creation
    #             assert num_nodes == data.graphs[i].number_of_nodes()
    #             node_types = torch.zeros((num_nodes, 3))
    #             node_types[1:num_var+1, 0] = 1
    #             node_types[num_var+1:num_var*2+1, 1] = 1
    #             node_types[num_var*2+1:, 2] = 1
    #             data.graphs[i].ndata['node_type'] = node_types
    #     return data
    
    # def _get_clause_pe(self, data, d_model: int = 10, max_len: int = 100000, scale_n = 10000):
    #     # https://pytorch.org/tutorials/beginner/transformer_tutorial.html#:~:text=PositionalEncoding%20module%20injects%20some%20information,cosine%20functions%20of%20different%20frequencies.
    #     position = torch.arange(max_len).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(scale_n) / d_model))
    #     pe = torch.zeros(max_len, d_model)
    #     pe[:, 0::2] = torch.sin(position * div_term)
    #     pe[:, 1::2] = torch.cos(position * div_term)

    #     d_nt = 3    # PE for node type. dim=3
    #     nt_div_term = torch.exp(torch.arange(0, d_nt, 2) * (-math.log(scale_n) / d_nt))
    #     nt_pe = torch.zeros(max_len, d_nt)
    #     nt_pe[:, 0::2] = torch.sin(position * nt_div_term)
    #     nt_pe[:, 1::2] = torch.cos(position * nt_div_term[0])
        
    #     for i, g in enumerate(data.graphs):
    #         if g.is_homogeneous:
    #             num_clause = data.labels[i][1]
    #             g.ndata['pe'][-num_clause:] += pe[:num_clause]
    #             g.ndata['node_type'][-num_clause:] += nt_pe[:num_clause]
    #         else:
    #             num_clause = g.ndata['_ID']['clause'].shape[0]
    #             g.ndata['pe']['clause'] += pe[:num_clause]
    #             g.ndata['node_type']['clause'] = g.ndata['node_type']['clause'].float() + nt_pe[:num_clause]
    #     return data

    def train_dataloader(self):
        return GraphDataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return GraphDataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return GraphDataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == '__main__':
    # Only for testing
    test_mode = 'train'

    if test_mode == 'train':
        datamodule = MiniShuffleDataModule()
        datamodule.setup('fit')
        train_dataloader = datamodule.train_dataloader()
        batch_0, label_0, info_0 = next(iter(train_dataloader))
        print(batch_0)
    elif test_mode == 'test':
        datamodule = MiniShuffleDataModule()
        datamodule.setup('test')
        test_dataloader = datamodule.test_dataloader()
        batch_0, label_0, info_0 = next(iter(test_dataloader))
        print(batch_0)