import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# label_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/runtimes_fixed.csv'
# feat_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc//satzilla_feats.csv'
# train_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/split_idx/train_idx.npy'
# val_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/split_idx/val_idx.npy'
# test_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_2k_bc/split_idx/test_idx.npy'


# label_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/runtimes_fixed_v9.csv'
# feat_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/satzilla_feats_v9.csv'
# train_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/split_idx/train_idx.npy'
# val_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/split_idx/val_idx.npy'
# test_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/w2sat_tseitin/split_idx/test_idx.npy'

label_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/runtimes_fixed_v9.csv'
feat_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/satzilla_feats_v9.csv'
train_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/split_idx/train_idx.npy'
val_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/split_idx/val_idx.npy'
test_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/tseitin_combined/split_idx/test_idx.npy'

# label_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/runtimes_fixed_v9.csv'
# feat_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/satzilla_feats_v9.csv'
# train_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/split_idx/train_idx.npy'
# val_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/split_idx/val_idx.npy'
# test_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/hardsatgen_tseitin/split_idx/test_idx.npy'



# label_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/runtimes.csv'
# feat_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/satzilla_feats.csv'
# train_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/split_idx/train_idx.npy'
# val_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/split_idx/val_idx.npy'
# test_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/split_idx/test_idx.npy'


# label_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/runtimes.csv'
# feat_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_combined/satzilla_feats.csv'
# train_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_og/split_idx/train_idx.npy'
# val_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_og/split_idx/val_idx.npy'
# test_idx_path = '/home/joseph-c/sat_gen/CoreDetection/training_exp/2k_bigcore_sz_og/split_idx/test_idx.npy'

class SATzillaDataset(Dataset):
    def __init__(self, feats, labels):
        super().__init__()
        self.feats = torch.tensor(feats, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __getitem__(self, i):
        # print(self.feats, self.labels)
        # print(self.feats.shape, self.labels.shape)
        return self.feats[i], self.labels[i], self.labels[i]   # Return 3 items to align with cnf_dataset

    def __len__(self):
        return len(self.feats)

class SATzillaDataModule():
    def __init__(self, batch_size: int = 32, train_val_split = [0.9, 0.1], num_workers = 32, debug = False, split_idx = 0, **kwargs) :
        super().__init__()
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.debug = debug
        self.split_idx = split_idx
        print('yippee -------------------')
    
    def setup(self, stage: str):
        df_rt_labels = pd.read_csv(label_path)
        rt_labels = df_rt_labels.iloc[:, 4:11].to_numpy().astype(float)

        # Load the satzilla features (Up to feature #33 in the Satzilla paper)
        df_sat_feats = pd.read_csv(feat_path)
        sat_feats = df_sat_feats.iloc[:, 1:].to_numpy()  # dim=[78730, 33]

        all_train_idx = np.load(train_idx_path, allow_pickle=True)
        all_val_idx = np.load(val_idx_path, allow_pickle=True)
        all_test_idx = np.load(test_idx_path, allow_pickle=True)

        train_idx = all_train_idx[self.split_idx]
        val_idx = all_val_idx[self.split_idx]
        test_idx = all_test_idx[self.split_idx]

        train_feats = sat_feats[train_idx]
        train_labels = rt_labels[train_idx]

        val_feats = sat_feats[val_idx]
        val_labels = rt_labels[val_idx]

        test_feats = sat_feats[test_idx]
        test_labels = rt_labels[test_idx]
        # print(train_feats)
        # print(train_labels)
        # print(train_idx)
        self.train_data = SATzillaDataset(train_feats, train_labels)
        self.val_data = SATzillaDataset(val_feats, val_labels)
        self.test_data = SATzillaDataset(test_feats, test_labels)
    
    def train_dataloader(self):
        return DataLoader(self.train_data,batch_size=self.batch_size,drop_last=False, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,batch_size=self.batch_size,drop_last=False, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,batch_size=self.batch_size,drop_last=False, shuffle=False)


if __name__ == '__main__':
    datamodule = SATzillaDataModule(split_idx=0)
    print('ff')