import torch
import pickle as pkl
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

label_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset/sc_1080_labels.pkl'
feat_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset/satzilla_feats.csv'

class SCSATzillaDataset(Dataset):
    def __init__(self, feats, labels):
        super().__init__()
        self.feats = torch.tensor(feats, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, i):
        return self.feats[i], self.labels[i], 0

    def __len__(self):
        return len(self.feats)

class SCSATzillaDataModule():
    # def __init__(self, batch_size: int = 32, use_balanced_idx=False, train_val_split = [0.9, 0.1], num_workers = 32, debug = False, split_idx = 0) :
    def __init__(self, sat_graph_type = 'hetero', use_balanced_idx=False, add_clause_pe = False, batch_size: int = 32, num_workers = 32, debug = False, split_idx = 0) :
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug
        self.split_idx = split_idx
        self.use_balanced_idx=use_balanced_idx
    
    def setup(self, stage: str):
        # df_rt_labels = pd.read_csv(label_path)
        # rt_labels = df_rt_labels.iloc[:, 4:11].to_numpy().astype(float)
        labels = pkl.load(open(label_path, 'rb')).numpy()

        # Load the satzilla features (Up to feature #33 in the Satzilla paper)
        df_sat_feats = pd.read_csv(feat_path)
        sat_feats = df_sat_feats.to_numpy()  # dim=[1080, 33]

        if self.use_balanced_idx:
            # train_idx_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset/sc_data_bal_2617_train_idx.npy'
            # test_idx_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset/sc_data_bal_2617_test_idx.npy'
            train_idx_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset/sc_data_balanced_train_idx.pkl'
            all_train_idx = pkl.load(open(train_idx_path, 'rb'))
        else:
            train_idx_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset/sc_data_1080_train_idx.npy'
            all_train_idx = np.load(train_idx_path)

        test_idx_path = '/home/vincent/sat/sat_selection_light/data/sc_dataset/sc_data_balanced_test_idx.pkl'
        all_test_idx = pkl.load(open(test_idx_path, 'rb'))

        train_idx = all_train_idx[self.split_idx]
        test_idx = all_test_idx[self.split_idx]

        test_idx = np.unique(test_idx)

        train_feats = sat_feats[train_idx]
        train_labels = labels[train_idx]

        test_feats = sat_feats[test_idx]
        test_labels = labels[test_idx]

        train_data = SCSATzillaDataset(train_feats, train_labels)
        self.train_data, self.val_data = random_split(train_data, [0.875, 0.125])
        self.test_data = SCSATzillaDataset(test_feats, test_labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data,batch_size=self.batch_size,drop_last=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,batch_size=self.batch_size,drop_last=False, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,batch_size=self.batch_size,drop_last=False, shuffle=False)


if __name__ == '__main__':
    datamodule = SCSATzillaDataModule(use_balanced_idx=True, split_idx=0)
    datamodule.setup('fit')
    train_dataloader = datamodule.train_dataloader()
    batch = next(iter(train_dataloader))
    print(batch)
    print('ff')