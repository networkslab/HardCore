import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import argparse
from infer_python.infer_trainer import Trainer

if __name__ == '__main__':
    batch_size = 1

    # Directory of CNF files to be inferred
    cnf_dir = '/home/vincent/sat/sat_selection/data/cnfs_1000/'

    # Configure file
    cfg_file = '/home/vincent/sat/sat_selection_light/configs/0912_pyg_6_solvers.yaml' 

    # Checkpoint path of trained pytorch model
    ckpt_path = '/home/vincent/sat/sat_selection_light/lightning_logs/0912_pyg_6_solvers/seed_604_split_0/checkpoints/best_epoch_67.pt'

    # Results will be saved under same experiment log directory 
    log_dir = ''.join(ckpt_path.split('.')[:-1])+'_infer'
    print(f'Log directory: {log_dir}')

    trainer = Trainer(cnf_dir, cfg_file, ckpt_path, log_dir, batch_size)
    trainer.infer(load_data=True)