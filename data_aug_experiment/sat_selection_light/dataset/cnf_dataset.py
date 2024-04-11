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

# This is minidata for debugging, as the full-scale data is too large
class NodezillaSATzillaMinidata(DGLDataset):
    def __init__(self, graph_type):
        # save_dir = '/home/vincent/sat/sat_selection/data/dgl_dataset/cnfs_1000_pe/minidata'
        save_dir = 'a;sdlkj;askdf'
        self.satzilla_feat_path = '/home/vincent/sat/baselines/data/satzilla/satzilla_feats.csv'
        if graph_type == 'hetero':
            self.graph_path = os.path.join(save_dir, 'minidata_heterographs.bin')
        elif graph_type == 'hetero_satzilla':
            self.graph_path = os.path.join(save_dir, 'minidata_heterographs_satzilla.bin')
        else:
            self.graph_path = os.path.join(save_dir, 'minidata_graphs.bin')
        self.info_path = os.path.join(save_dir, 'minidata_info.pkl')
        super().__init__(name='cnfs_mini', save_dir=save_dir)
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i].float(), self.satzilla_feat[i].float()
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        return

    def load(self):
        # load processed data from directory `self.save_path`
        self.satzilla_feat = torch.tensor(pd.read_csv(self.satzilla_feat_path).iloc[:, 1:].to_numpy())
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.labels = label_dict['label']
        # self.graph_info = load_info(self.info_path)

    def has_cache(self):
        if not os.path.exists(self.graph_path):
            raise ValueError(f"The dataset cannot be found at: {self.graph_path}")
        return True 

class NodezillaSATzillaDataset(DGLDataset):
    '''
    This class import the whole dataset.
    '''
    def __init__(self, graph_type, idx_list=None):
        # save_dir = '/home/vincent/sat/sat_selection/data/dgl_dataset/cnfs_1000_pe'
        save_dir = 'asdfasdf'
        self.satzilla_feat_path = '/home/vincent/sat/baselines/data/satzilla/satzilla_feats.csv'
        if graph_type == 'hetero':
            self.graph_path = os.path.join(save_dir, 'all_dataset_hetero.bin')
        elif graph_type == 'hetero_satzilla':
            self.graph_path = os.path.join(save_dir, 'all_dataset_hetero_satzilla.bin')
        else:
            self.graph_path = os.path.join(save_dir, 'all_dataset.bin')
        self.info_path = os.path.join(save_dir, 'all_info.pkl')
        self.idx_list = idx_list
        super().__init__(name='cnfs_1000', save_dir=save_dir)
    
    def process(self):
        return
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i].float(), self.satzilla_feat[i].float()
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        return

    def load(self):
        # load processed data from directory `self.save_path`
        print("Loading data...")
        if self.idx_list is None:
            self.satzilla_feat = torch.tensor(pd.read_csv(self.satzilla_feat_path).iloc[:, 1:].to_numpy())
            self.graphs, label_dict = load_graphs(self.graph_path)
            self.labels = label_dict['label']
            # self.graph_info = load_info(self.info_path)
            # self.graph_info = [-1] * len(self.graphs)   # Comment out graph_info as loading is too slow
        else:
            idx_list = list(map(int, self.idx_list))   # For some reason have to manually map the index to int otherwise load_graphs doesn't work
            # graph_info = pkl.load(open(self.info_path, 'rb'))
            # self.graph_info = graph_info[idx_list]
            self.satzilla_feat = torch.tensor(pd.read_csv(self.satzilla_feat_path).iloc[:, 1:].to_numpy())
            self.graphs, label_dict = load_graphs(self.graph_path, idx_list=idx_list)
            self.labels = label_dict['label'][idx_list]
            # self.graph_info = [-1] * len(idx_list)
        print(f"Dataset has been loaded. Length: {len(self.graphs)}")

    def has_cache(self):
        if not os.path.exists(self.graph_path):
            raise ValueError(f"The dataset cannot be found at: {self.graph_path}")
        return True 

class CNF1000Dataset(DGLDataset):
    '''
    This class import the whole dataset.
    '''
    def __init__(self, graph_type, idx_list=None):
        save_dir = '/home/joseph-c/sat_gen/CoreDetection/training_exp/5k_bigcore_og/cnfs_1000_pe'
        if graph_type == 'hetero':
            self.graph_path = os.path.join(save_dir, 'all_dataset_hetero.bin')
        elif graph_type == 'hetero_satzilla':
            self.graph_path = os.path.join(save_dir, 'all_dataset_hetero_satzilla.bin')
        elif graph_type == 'nodezilla_gcn':
            self.graph_path = os.path.join(save_dir, 'all_dataset_nodezilla.bin')
        else:
            self.graph_path = os.path.join(save_dir, 'all_dataset.bin')
        self.info_path = os.path.join(save_dir, 'all_info.pkl')
        self.idx_list = idx_list
        super().__init__(name='cnfs_1000', save_dir=save_dir)
    
    def process(self):
        return
    
    def __getitem__(self, i):
        # print(i)
        # print(len(self.graphs), len(self.labels), len(self.graph_info))
        return self.graphs[i], self.labels[i].float(), self.graph_info[i]
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        return

    def load(self):
        # load processed data from directory `self.save_path`
        print("Loading data...")
        if self.idx_list is None:
            self.graphs, label_dict = load_graphs(self.graph_path)
            self.labels = label_dict['label']
            # self.graph_info = load_info(self.info_path)
            self.graph_info = [-1] * len(self.graphs)   # Comment out graph_info as loading is too slow
        else:
            idx_list = list(map(int, self.idx_list))   # For some reason have to manually map the index to int otherwise load_graphs doesn't work
            # graph_info = pkl.load(open(self.info_path, 'rb'))
            # self.graph_info = graph_info[idx_list]
            self.graphs, label_dict = load_graphs(self.graph_path, idx_list=idx_list)
            self.labels = label_dict['label'][idx_list]
            self.graph_info = [-1] * len(idx_list)
        if self.labels.shape[1] > 7:
            runtimes = self.labels[:, 2:-1]
        else:
            runtimes = self.labels
        top1_idx = runtimes.argmin(dim=-1)
        label_cnt = torch.tensor([(top1_idx==i).sum() for i in range(7)])
        self.label_ratio = label_cnt / label_cnt.sum()

        print(f"Dataset has been loaded. Length: {len(self.graphs)}")

    def has_cache(self):
        if not os.path.exists(self.graph_path):
            raise ValueError(f"The dataset cannot be found at: {self.graph_path}")
        return True 

# Run process_dataset.py first to generate different splits of the data
class CNF1000DatasetSplit(DGLDataset):
    def __init__(self, train=True, split_idx=0):
        self.train = train
        self.split_idx = split_idx
        save_dir = '/home/vincent/sat/sat_selection/data/dgl_dataset/cnfs_1000_pe'
        if self.train:
            self.graph_path = os.path.join(save_dir, 'split_'+str(split_idx), 'train_dataset.bin')
            self.info_path = os.path.join(save_dir, 'split_'+str(split_idx), 'train_info.pkl')
        else:
            self.graph_path = os.path.join(save_dir, 'split_'+str(split_idx), 'test_dataset.bin')
            self.info_path = os.path.join(save_dir, 'split_'+str(split_idx), 'test_info.pkl')
        super().__init__(name='cnfs_amur', save_dir=save_dir)
    
    def process(self):
        return
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i].float(), self.graph_info[i]
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        return

    def load(self):
        # load processed data from directory `self.save_path`
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.labels = label_dict['label']
        self.graph_info = load_info(self.info_path)
        print(f"Split_{self.split_idx} dataset has been loaded. Length: {len(self.graphs)}")
        # print(f"Name of first data: {self.graph_info[0]['cnf_name']}")

    def has_cache(self):
        if not os.path.exists(self.graph_path):
            raise ValueError(f"The dataset cannot be found at: {self.graph_path}")
        return True 

# This is minidata for debugging, as the full-scale data is too large
class CNF1000Minidata(DGLDataset):
    def __init__(self, graph_type):
        # save_dir = '/home/vincent/sat/sat_selection/data/dgl_dataset/cnfs_1000_pe/minidata'
        save_dir = '/asdfasdfasf'
        if graph_type == 'hetero':
            self.graph_path = os.path.join(save_dir, 'minidata_heterographs.bin')
        elif graph_type == 'hetero_satzilla':
            self.graph_path = os.path.join(save_dir, 'minidata_heterographs_satzilla.bin')
        elif graph_type == 'nodezilla_gcn':
            self.graph_path = os.path.join(save_dir, 'minidata_graphs_nodezilla.bin')
        else:
            self.graph_path = os.path.join(save_dir, 'minidata_graphs.bin')
        self.info_path = os.path.join(save_dir, 'minidata_info.pkl')
        super().__init__(name='cnfs_mini', save_dir=save_dir)
    
    def __getitem__(self, i):
        return self.graphs[i], self.labels[i].float(), self.graph_info[i]
    
    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        return

    def load(self):
        # load processed data from directory `self.save_path`
        self.graphs, label_dict = load_graphs(self.graph_path)
        self.labels = label_dict['label']
        self.graph_info = load_info(self.info_path)
        if self.labels.shape[1] > 7:
            runtimes = self.labels[:, 2:-1]
        else:
            runtimes = self.labels
        top1_idx = runtimes.argmin(dim=-1)
        label_cnt = torch.tensor([(top1_idx==i).sum() for i in range(7)])
        self.label_ratio = label_cnt / label_cnt.sum()


    def has_cache(self):
        if not os.path.exists(self.graph_path):
            raise ValueError(f"The dataset cannot be found at: {self.graph_path}")
        return True 

class CNF1000DataModule():
    def __init__(self, sat_data='gnn', sat_graph_type = 'hetero', add_clause_pe = False, use_balanced_idx=False, batch_size: int = 32, num_workers = 32, debug = False, split_idx = 0) :
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.debug = debug
        self.debug = False
        self.split_idx = split_idx
        self.sat_data = sat_data
        self.graph_type = sat_graph_type
        self.add_clause_pe = add_clause_pe
        self.use_balanced_idx = use_balanced_idx
    
    def setup(self, stage: str):
        # Split the data. Transformation
        if self.debug is True:
            if self.sat_data == 'nodezilla_satzilla':
                minidata = NodezillaSATzillaMinidata(self.graph_type)
            else:
                minidata = CNF1000Minidata(self.graph_type)
            self.label_ratio = minidata.label_ratio
            minidata = self._get_node_type(minidata)
            # if (self.graph_type == 'hetero') and self.add_clause_pe:
            if self.add_clause_pe:
                minidata = self._get_clause_pe(minidata)
            self.train_data, self.val_data, self.test_data = split_dataset(minidata, [0.4, 0.3, 0.3])
            return

        # Use pre-generated data split
        if self.use_balanced_idx:
            # train_idx_path = '/home/vincent/sat/sat_selection/data/split_idx/train_idx_bal.pkl'
            train_idx_path = 'asdfasdf'
            all_train_idx = pkl.load(open(train_idx_path, 'rb'))
        else:
            train_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/5k_bigcore_og/split_idx/train_idx.npy'
            all_train_idx = np.load(train_idx_path, allow_pickle=True)

        train_idx = all_train_idx[self.split_idx]

        val_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/5k_bigcore_og/split_idx/val_idx.npy'
        test_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/5k_bigcore_og/split_idx/test_idx.npy'

        all_val_idx = np.load(val_idx_path, allow_pickle=True)
        all_test_idx = np.load(test_idx_path, allow_pickle=True)

        val_idx = all_val_idx[self.split_idx]
        test_idx = all_test_idx[self.split_idx]
        
        if stage == 'test':
            test_data = CNF1000Dataset(self.graph_type, list(test_idx))
            if self.add_clause_pe:
                test_data = self._get_clause_pe(test_data)
            self.test_data = test_data
        else:
            if self.sat_data == 'nodezilla_satzilla':
                all_data = NodezillaSATzillaDataset(self.graph_type)
            else:
                all_data = CNF1000Dataset(self.graph_type)
            self.label_ratio = all_data.label_ratio
            all_data = self._get_node_type(all_data)
            # if (self.graph_type == 'hetero') and self.add_clause_pe:
            if self.add_clause_pe:
                all_data = self._get_clause_pe(all_data)

            self.train_data = Subset(all_data, list(train_idx)) 
            self.val_data = Subset(all_data, list(val_idx)) 
            self.test_data = Subset(all_data, list(test_idx)) 
    
    def _get_node_type(self, data):
        # Same node type used by Wenyi. [pos_literal, neg_literal, clause]
        if not data.graphs[0].is_homogeneous:  # Heterogeneous graph
            for i in range(len(data)):
                hg = data.graphs[i]
                num_lit = hg.ndata['_ID']['pos_lit'].shape[0]
                num_clause = hg.ndata['_ID']['clause'].shape[0]
                data.graphs[i].nodes['pos_lit'].data['node_type'] = F.one_hot(torch.zeros((num_lit), dtype=torch.long), num_classes=3)
                data.graphs[i].nodes['neg_lit'].data['node_type'] = F.one_hot(torch.ones((num_lit), dtype=torch.long), num_classes=3)
                data.graphs[i].nodes['clause'].data['node_type'] = F.one_hot(2*torch.ones((num_clause), dtype=torch.long), num_classes=3)
 
                # data.graphs[i].ndata['onehot']['pos_lit'] = F.one_hot(torch.zeros((num_lit), dtype=torch.long), num_classes=3)
                # data.graphs[i].ndata['onehot']['neg_lit'] = F.one_hot(torch.ones((num_lit), dtype=torch.long), num_classes=3)
                # data.graphs[i].ndata['onehot']['clause'] = F.one_hot(2*torch.ones((num_clause), dtype=torch.long), num_classes=3)
        else:
            for i in range(len(data)):
                # num_var = data.graph_info[i]['num_var']
                # num_clause = data.graph_info[i]['num_clause']
                num_var = data.labels[i][0]
                num_clause = data.labels[i][1]
                num_nodes = num_var * 2 + num_clause  # The node_0 is dumb node existed during graph creation
                assert num_nodes == data.graphs[i].number_of_nodes()
                node_types = torch.zeros((num_nodes, 3))
                node_types[:num_var, 0] = 1
                node_types[num_var:num_var*2, 1] = 1
                node_types[num_var*2:, 2] = 1
                data.graphs[i].ndata['node_type'] = node_types
        return data
    
    def _get_clause_pe(self, data, d_model: int = 10, max_len: int = 100000, scale_n = 10000):
        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html#:~:text=PositionalEncoding%20module%20injects%20some%20information,cosine%20functions%20of%20different%20frequencies.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(scale_n) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        d_nt = 3    # PE for node type. dim=3
        nt_div_term = torch.exp(torch.arange(0, d_nt, 2) * (-math.log(scale_n) / d_nt))
        nt_pe = torch.zeros(max_len, d_nt)
        nt_pe[:, 0::2] = torch.sin(position * nt_div_term)
        nt_pe[:, 1::2] = torch.cos(position * nt_div_term[0])
        
        for i, g in enumerate(data.graphs):
            if g.is_homogeneous:
                num_clause = data.labels[i][1]
                g.ndata['pe'][-num_clause:] += pe[:num_clause]
                g.ndata['node_type'][-num_clause:] += nt_pe[:num_clause]
            else:
                num_clause = g.ndata['_ID']['clause'].shape[0]
                # Node data cannot be assigned with g.ndata. Have assign in nodes[]
                # Clause PE for node-level satzilla is included in encoder.py 
                # g.ndata['pe']['clause'] += pe[:num_clause]
                # g.ndata['node_type']['clause'] = g.ndata['node_type']['clause'].float() + nt_pe[:num_clause]
                g.nodes['clause'].data['pe'] += pe[:num_clause]
                g.nodes['clause'].data['node_type'] = g.nodes['clause'].data['node_type'].float() + nt_pe[:num_clause]

                # node_satzilla = g.nodes['clause'].data['satzilla']
                # node_satzilla_padded = torch.concat((node_satzilla, torch.zeros((num_clause, 3))), dim=1)
                # g.nodes['clause'].data['satzilla'] = node_satzilla_padded + pe[:num_clause]
        return data

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
        datamodule = CNF1000DataModule()
        datamodule.setup('fit')
        train_dataloader = datamodule.train_dataloader()
        batch_0, label_0, info_0 = next(iter(train_dataloader))
        print(batch_0)
    elif test_mode == 'test':
        datamodule = CNF1000DataModule()
        datamodule.setup('test')
        test_dataloader = datamodule.test_dataloader()
        batch_0, label_0, info_0 = next(iter(test_dataloader))
        import pickle as pkl
        with open('batch_0.pkl', 'wb') as f:
            pkl.dump(batch_0, f)
        with open('label_0.pkl', 'wb') as f:
            pkl.dump(label_0, f)
        with open('info_0.pkl', 'wb') as f:
            pkl.dump(info_0, f)
        print(batch_0)