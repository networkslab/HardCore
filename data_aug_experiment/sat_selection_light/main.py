import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import argparse
from trainer.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('subcommand') 
    parser.add_argument('-c', '--config', default='/home/joseph-c/sat_gen/CoreDetection/training_exp/sat_selection_light/configs/0604_satzilla_mlp.yaml')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--version', type=str, default='sat_reg_unleaky', help='For log directory folder.')
    parser.add_argument('--device', type=int, default=0, help='GPU ID.')
    parser.add_argument('--split_idx', type=int, default=0, help='Split index of dataset.')
    parser.add_argument('--debug', action='store_false')
    # parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--run_test', action='store_true')

    args = parser.parse_args()
    
    trainer = Trainer(args)

    trainer.fit()
    trainer.test()
