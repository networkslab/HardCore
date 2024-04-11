import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import argparse
from trainer.pyg_trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('subcommand') 
    parser.add_argument('-c', '--config', default='configs/0912_pyg_6_solvers.yaml')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--version', type=str, default=None, help='For log directory folder.')
    parser.add_argument('--device', type=int, default=0, help='GPU ID.')
    parser.add_argument('--split_idx', type=int, default=0, help='Split index of dataset.')
    parser.add_argument('--debug', action='store_false')
    # parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--run_test', action='store_true')

    args = parser.parse_args()
    
    trainer = Trainer(args)

    if args.run_test:
        trainer.test(load_data=True)
    else:
        trainer.fit()
        trainer.test()
