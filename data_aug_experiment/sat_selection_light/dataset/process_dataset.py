import os
import numpy as np
import random
from tqdm import tqdm
import time

from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix, linalg
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
import pandas as pd
import dgl
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info, split_dataset

def split_dataset():
    dataset_dir = 'data/dgl_dataset/cnfs_amur/'

    dataset_path = os.path.join(dataset_dir, 'full_bigger.bin')

    graphs, label_dict = load_graphs(dataset_path)
    labels = label_dict['label']

    n = len(graphs)
    n_test = int(n*0.2)
    idx = np.random.RandomState(seed=0).permutation(n)

    k = 5
    for i in range(k):
        save_path = os.path.join(dataset_dir, 'split_'+str(i))
        os.makedirs(save_path, exist_ok=True)

        test_idx = idx[n_test*i:n_test*(i+1)]
        test_data = [graphs[j] for j in test_idx]
        test_label = labels[test_idx]
        test_path = os.path.join(save_path, 'test_dataset.bin')
        save_graphs(test_path, test_data, {'label': test_label})

        train_idx = np.delete(idx, test_idx)
        train_data = [graphs[j] for j in train_idx]
        train_label = labels[train_idx]
        train_path = os.path.join(save_path, 'train_dataset.bin')
        save_graphs(train_path, train_data, {'label': train_label})

class ProcessDataset():
    def __init__(self, name='cnfs_1000_pe', k_fold=5, pe_dim=10, dataset_dir='/home/vincent/sat/sat_selection/data/dgl_dataset'):
        super().__init__()
        self.name = name
        self.pe_dim = pe_dim
        self.mtx_dir = 'data/mtx'
        self.label_filename = 'data/cnfs_1000_results_0220.csv'
        self.save_path = os.path.join(dataset_dir, name)

        if not self.has_cache():
            self.process_cnf_dataset()
        else:
            self.load_dataset()
        
        if k_fold > 1:
            self.k_fold_split(k_fold)

    def process_cnf_dataset(self):
        self.graphs = []
        self.labels = []
        self.graph_info = []

        df_label = pd.read_csv(self.label_filename)

        label_names = ['#var', '#clause', 'base', 'HyWalk', 'MOSS', 'mabgb', 'ESA', 'bulky', 'UCB', 'MIN']
        len_data = df_label.shape[0]
        total_time_pe = 0
        for i in tqdm(range(len_data)):
            name = df_label['name'][i]
            mtx_filename = name.split('.')[0]+'.mtx'
            mtx_file = os.path.join(self.mtx_dir, mtx_filename)
            mm = mmread(mtx_file)
            g = dgl.from_scipy(mm)

            # Convert to bidirected graph
            g = dgl.to_bidirected(g)
            t0 = time.time()
            pe = self._add_undirected_graph_positional_embedding(g, self.pe_dim)
            t_pe = time.time() - t0
            print(f"{i}:\tPE: {t_pe}")
            total_time_pe += t_pe

            # Node degrees upto 4 hops
            node_degree = torch.zeros(g.num_nodes(), 4)
            for k in range(1, 5):
                adj = dgl.khop_adj(g, k)
                node_degree[:, k-1] = adj.sum(dim=1) 

            # Node type: {variable: 0, clause: 1}.
            node_types = torch.zeros(g.num_nodes())
            num_clause = df_label['#clause'][i]
            node_types[-num_clause:] = 1

            # Labels
            label = [df_label[n][i] for n in label_names]

            g.ndata['pe'] = pe 
            g.ndata['node_deg'] = node_degree
            g.ndata['node_type'] = node_types

            # Get graph stats inspired by SATzilla
            graph_stats = self.get_graph_stats(g, df_label.iloc[i])
            graph_stats['time_pe'] = t_pe # Add time to compute pe
            graph_stats['cnf_name'] = name # Add cnf name for potential use

            self.graphs.append(g)
            self.labels.append(label)
            self.graph_info.append(graph_stats)
        self.labels = torch.tensor(self.labels)
        print(f"Total time to compute PE: {total_time_pe}")

        # Save the dataset
        graph_path = os.path.join(self.save_path, 'all_dataset.bin')
        save_graphs(graph_path, self.graphs, {'label': self.labels})
        # save other information in python dict
        info_path = os.path.join(self.save_path, 'all_info.pkl')
        save_info(info_path, self.graph_info)
        return

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'all_dataset.bin')
        info_path = os.path.join(self.save_path, 'all_info.pkl')
        if os.path.exists(graph_path) and os.path.exists(info_path):
            print('Found dataset. Loading...')
            return True
        else:
            print('Cannot find dataset. Processing...')
            return False

    def load_dataset(self):
        graph_path = os.path.join(self.save_path, 'all_dataset.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['label']
        info_path = os.path.join(self.save_path, 'all_info.pkl')
        self.graph_info = load_info(info_path)
        print('Dataset loaded successfully!')

    def get_graph_stats(self, g, label_i):
        # Compute LCG features
        num_clause = label_i['#clause']
        num_var = label_i['#var']
        clause_var_ratio = num_clause / num_var

        # Compute degree features of var node
        deg_all = g.in_degrees()
        deg_var  = deg_all[:num_var].float()
        deg_var_mean = deg_var.mean()
        deg_var_var = deg_var.var()
        deg_var_min = deg_var.min()
        deg_var_max = deg_var.max()
        _, _cnt = deg_var.unique(return_counts=True)
        _prob = _cnt / _cnt.sum()
        deg_var_entropy = -torch.sum(_prob * torch.log(_prob))

        # Compute degree features of clause nodes
        deg_clause  = deg_all[-num_clause:].float()
        deg_clause_mean = deg_clause.mean()
        deg_clause_var = deg_clause.var()
        deg_clause_min = deg_clause.min()
        deg_clause_max = deg_clause.max()
        _, _cnt = deg_clause.unique(return_counts=True)
        _prob = _cnt / _cnt.sum()
        deg_clause_entropy = -torch.sum(_prob * torch.log(_prob))

        graph_stats = {
            'num_clause': num_clause,
            'num_var': num_var,
            'clause_var_ratio': clause_var_ratio,
            'deg_var_mean': deg_var_mean,
            'deg_var_var': deg_var_var,
            'deg_var_min': deg_var_min,
            'deg_var_max': deg_var_max,
            'deg_var_entropy': deg_var_entropy,
            'deg_clause_mean': deg_clause_mean,
            'deg_clause_var': deg_clause_var,
            'deg_clause_min': deg_clause_min,
            'deg_clause_max': deg_clause_max,
            'deg_clause_entropy': deg_clause_entropy}
        return graph_stats

    def eigen(self, n, k, laplacian, hidden_size, retry):
        if k <= 0:
            return torch.zeros(n, hidden_size)
        laplacian = laplacian.astype("float64")
        ncv = min(n, max(2 * k + 1, 20))
        v0 = np.random.rand(n).astype("float64")
        for i in range(retry):
            try:
                s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
            except sparse.linalg.eigen.arpack.ArpackError:
                # print("arpack error, retry=", i)
                ncv = min(ncv * 2, n)
                if i + 1 == retry:
                    sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                    u = torch.zeros(n, k)
            else:
                break
        x = preprocessing.normalize(u, norm="l2")
        sign_x = np.sign(x[0,:])

        x = np.transpose(np.transpose(x) * sign_x[:,None])
        x = torch.from_numpy(x.astype("float32"))

        x = F.pad(x, (0, hidden_size - k), "constant", 0)
        return x

    def _add_undirected_graph_positional_embedding(self, g, hidden_size, retry=10):
        # We use eigenvectors of normalized graph laplacian as vertex features.
        # It could be viewed as a generalization of positional embedding in the
        # attention is all you need paper.
        # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
        # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
        n = g.number_of_nodes()
        adj = g.adjacency_matrix(transpose=False,  scipy_fmt="csr").astype(float)
        norm = sparse.diags(
            dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
        )
        laplacian = norm * adj * norm
        k = min(n - 2, hidden_size)
        x = self.eigen(n, k, laplacian, hidden_size, retry)
        return x.float()
    
    def k_fold_split(self, k):
        n = len(self.graphs)
        n_test = int(n*0.2)
        idx = np.random.RandomState(seed=0).permutation(n)

        print('Generating K-fold split.')
        k = 5
        for i in tqdm(range(k)):
            split_save_path = os.path.join(self.save_path, 'split_'+str(i))
            os.makedirs(split_save_path, exist_ok=True)

            # Save test data, label and graph_info
            test_idx = idx[n_test*i:n_test*(i+1)]
            test_data = [self.graphs[j] for j in test_idx]
            test_label = self.labels[test_idx]
            test_path = os.path.join(split_save_path, 'test_dataset.bin')
            save_graphs(test_path, test_data, {'label': test_label})
            test_info = [self.graph_info[j] for j in test_idx]
            test_info_path = os.path.join(split_save_path, 'test_info.pkl')
            save_info(test_info_path, test_info)

            # Save train data, label and graph_info
            train_idx = np.delete(idx, test_idx)
            train_data = [self.graphs[j] for j in train_idx]
            train_label = self.labels[train_idx]
            train_path = os.path.join(split_save_path, 'train_dataset.bin')
            save_graphs(train_path, train_data, {'label': train_label})
            train_info = [self.graph_info[j] for j in train_idx]
            train_info_path = os.path.join(split_save_path, 'train_info.pkl')
            save_info(train_info_path, train_info)
        print('Finished k-fold.')

if __name__ == '__main__':
    ProcessDataset(k_fold=5)