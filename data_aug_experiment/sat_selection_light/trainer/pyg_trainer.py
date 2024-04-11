import os
import yaml
import random
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
from torch.optim import lr_scheduler
# from pl_module.sat_selection_cross_attn import SATSolverModule
# from pl_module.sat_selection import SATSolverModule
from pl_module.pyg_sat_selection import SATSolverModule
from dataset.cnf_dataset import CNF1000DataModule
from dataset.satzilla import SATzillaDataModule 

from .logger import Logger

class Trainer():
    def __init__(self, args):
        cfg_file = args.config
        self.cfg = yaml.safe_load(open(cfg_file, 'r'))
        # Update the parsed arguments from command line
        self.cfg['seed_everything'] = args.seed
        self.cfg['trainer']['logger'][0]['init_args']['version'] = args.version
        self.cfg['trainer']['devices'] = [args.device]
        self.cfg['data']['split_idx'] = args.split_idx
        self.cfg['data']['debug'] = args.debug
        self.cfg['model']['add_clause_pe'] = self.cfg['data']['add_clause_pe']

        self.logger = Logger(**self.cfg['trainer']['logger'][0]['init_args'])

        seed = self.cfg['seed_everything']
        self.seed_everything(seed)
        self.init_nn(self.cfg)
        self.max_epochs = self.cfg['trainer']['max_epochs']
        
        self.early_stop_patience = self.cfg['trainer']['callbacks'][1]['init_args']['patience']
        self.best_epoch = 0
        self.best_val_loss = np.inf

        yaml.dump(self.cfg, open(os.path.join(self.logger.log_dir, 'config.yaml'), 'w'))

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_nn(self, cfg):
        if cfg['sat_data'] == 'satzilla':
            self.data = SATzillaDataModule(**cfg['data'])
        else:
            self.data = CNF1000DataModule(**cfg['data'])

        self.model = SATSolverModule(cfg['model'])
        print(self.model)
        pkl.dump(cfg['model'], open(os.path.join(self.logger.log_dir, 'model_cfg.pkl'), 'wb'))
        # print(f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}")
        self.model.logger = self.logger
        self.optimizer = self.model.configure_optimizers()
        self.lr_scheduler = None
        if cfg['model']['lr_scheduler_flag'] is True:
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, cfg['trainer']['max_epochs'])
        self.set_device(cfg)

    def set_device(self, cfg):
        if torch.cuda.is_available() and cfg['trainer']['accelerator'] == 'gpu':
            self.device = 'cuda:' + str(cfg['trainer']['devices'][0])
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.device = self.device

    def transfer_batch_to_device(self, batch):
        # graphs, labels, graph_infos = batch
        batch[0] = batch[0].to(self.device)
        batch[1] = batch[1].to(self.device)

    def fit(self):
        # The main function for training
        self.data.setup('fit')
        train_dataloader = self.data.train_dataloader()
        val_dataloader = self.data.val_dataloader()
        
        self.model.train()
        torch.set_grad_enabled(True)

        for num_epoch in range(self.max_epochs):
            print(f"Epoch: {num_epoch} Training...")
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                self.transfer_batch_to_device(batch) 
                loss = self.model.training_step(batch, batch_idx)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print(f"Evaluation...")
            self.model.eval()
            with torch.no_grad():
                validation_step_outputs = []
                for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                    self.transfer_batch_to_device(batch) 
                    step_out = self.model.validation_step(batch, batch_idx)
                    validation_step_outputs.append(step_out)
                self.model.validation_epoch_end(validation_step_outputs)

                # Compute epoch-level metrics
                self.logger.on_epoch_end()
                
                if self.logger.epoch_metrics['val_loss'][-1] < self.best_val_loss:
                    self.save_ckpt(num_epoch, 'best')
                    self.best_val_loss = self.logger.epoch_metrics['val_loss'][-1]
                    self.best_epoch = num_epoch
                
                if (num_epoch - self.best_epoch) > self.early_stop_patience:
                    break
            
            if self.lr_scheduler != None:
                self.lr_scheduler.step()
                print(f'---Learning rate: {self.lr_scheduler.get_last_lr()}')
        
        self.save_ckpt(num_epoch, 'last')
        self.logger.save_metrics()

    def test(self, load_data=False):
        # Test function with the model with best val_loss
        self.load_ckpt()

        if load_data:
            self.data.setup('test')
        test_dataloader = self.data.test_dataloader()
        
        self.model.eval()
        torch.set_grad_enabled(False)

        with torch.no_grad():
            test_step_outputs = []
            for batch_idx, batch in enumerate(test_dataloader):
                self.transfer_batch_to_device(batch) 
                step_out = self.model.test_step(batch, batch_idx)
                test_step_outputs.append(step_out)
            self.model.test_epoch_end(test_step_outputs)

    def save_ckpt(self, num_epoch, mode):
        ckpt_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        if mode == 'best':  # Remove existing models
            old_files = [os.path.join(ckpt_dir, fname) for fname in os.listdir(ckpt_dir)]
            for f in old_files:
                os.remove(f)
        ckpt_name = f"{mode}_epoch_{num_epoch}.pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        torch.save(self.model.state_dict(), ckpt_path)

    def load_ckpt(self):
        ckpt_dir = os.path.join(self.logger.log_dir, 'checkpoints')
        ckpt_name = [n for n in os.listdir(ckpt_dir) if n.startswith('best')][0]
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.model.load_state_dict(torch.load(ckpt_path))