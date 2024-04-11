import os
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor

import dgl
import dgl.nn.pytorch as dglnn
import pandas as pd

# import pytorch_lightning as pl
from model.encoder.encoder import  SATzillaEncoder
from model.decoder.decoder import ClassifierDecoder


# from .metric.metric import HaltRate, AvgRuntime
# from pytorch_lightning.callbacks import BaseFinetuning

class SATSolverModule(nn.Module):
    # def __init__(self, sat_encoder: nn.Module, solver_encoder: nn.Module, cls_decoder: nn.Module, solver_enc_mode='none', aux_task=None, aux_loss_coef=1000, pca_mode='graph_7', lr=0.001):
    def __init__(self, cfg):
        super().__init__()
        sat_enc_init_args = cfg['sat_encoder']['init_args']
        sat_enc_init_args['feat_type'] = cfg['sat_feat_type']
        sat_enc_init_args['add_clause_pe'] = cfg['add_clause_pe']
        if cfg['sat_data'] == 'satzilla':
            self.sat_encoder = SATzillaEncoder(33)
        
        self.sat_data = cfg['sat_data']

        
        self.cls_decoder = ClassifierDecoder(**cfg['cls_decoder']['init_args'])

        self.num_classes = self.cls_decoder.num_classes
        
        self.sat_feat_type = cfg['sat_feat_type']
        self.solver_enc_mode = cfg['solver_enc_mode']
        self.aux_task = cfg['aux_task']
        self.aux_loss_coef = cfg['aux_loss_coef']
        self.optim_type = cfg['optim_type']
        pca_mode = cfg['pca_mode']
        

        if self.aux_task:
            self.aux_loss_func = nn.BCELoss()
        self.lr = float(cfg['lr'])
        self.cfg = cfg

    def set_loss_fn(self, loss_weights=None):
        self.loss_type = self.cfg['loss_type']
        if self.loss_type == 'ce':
            if loss_weights != None:
                self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward_step(self, batch):
        sat_graph, labels, graph_info = batch

        if isinstance(sat_graph, torch.Tensor):   # If uses SATzilla features instead of SAT graphs
            sat_h = self.sat_encoder(sat_graph)
        else:
            if self.sat_feat_type == 'pe':
                sat_feat = sat_graph.ndata['pe']
            elif self.sat_feat_type == 'satzilla':
                sat_feat = sat_graph.ndata['satzilla']
           

            sat_h = self.sat_encoder(sat_graph, sat_feat)

        solver_h = None
        
        # Concat satzilla feature with GNN output
        

        logits = self.cls_decoder(sat_h, solver_h)
        return logits

    def training_step(self, batch, batch_idx):
        # labels in dataset includes the following info: ['#var', '#clause', 'base', 'HyWalk', 'MOSS', 'ESA', 'bulky', 'UCB', 'MIN'] 
        # Note that g is batched graphs
        sat_graph, labels, graph_info = batch
        if labels.shape[1] == 12:
            runtime = labels[:, 4:-1]
        if labels.shape[1] == 10:
            runtime = labels[:, 2:-1]   # [#var, #clause, runtime_solvers, min_runtime]
        else:
            runtime = labels
            
        # Remove ESA
        # if self.num_classes == 6:
        #     runtime = torch.concat([runtime[:, :4], runtime[:, 5:]], dim=-1)
        if self.num_classes == 5:   # Remove kissat3 and bulky
            runtime = torch.concat([runtime[:, 1:5], runtime[:, 6:]], dim=-1)

        min_runtime, label_idx = runtime.min(dim=1)

        bs = len(labels) 

        # Call forward step
        logits = self.forward_step(batch)

        # Cross Entropy loss or Expected runtime loss
        if self.loss_type == 'ce':
            train_loss = self.loss_fn(logits, label_idx)
        elif self.loss_type == 'regression':
            # Masked out the samples greater or equal than 1450
            mask = runtime.le(1450)
            runtime_masked = torch.masked_select(runtime, mask)
            logits_masked = torch.masked_select(logits, mask)
            train_loss = F.mse_loss(logits_masked, runtime_masked)
        else: # Expected runtime loss
            probs = F.normalize(torch.sigmoid(logits), p=1)
            exp_runtime = torch.sum(torch.mul(probs, runtime), dim=1)
            train_loss = F.mse_loss(exp_runtime, min_runtime) / bs

        if self.aux_task:
            aux_loss = self.aux_loss * self.aux_loss_coef
            train_loss += aux_loss 
            self.log('aux_loss', aux_loss)
        self.log('train_loss', train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        sat_graph, labels, graph_info = batch
        # print(labels.shape)
        if labels.shape[1] == 12:
            runtime = labels[:, 4:-1]
        if labels.shape[1] == 10:
            runtime = labels[:, 2:-1]   # [#var, #clause, runtime_solvers, min_runtime]
        else:
            runtime = labels

        if self.num_classes == 5:   # Remove kissat3 and bulky
            runtime = torch.concat([runtime[:, 1:5], runtime[:, 6:]], dim=-1)

        # print(labels)
        min_runtime, label_idx = runtime.min(dim=1)

        bs = len(labels) 

        logits = self.forward_step(batch)

        # Cross Entropy loss or Expected runtime loss
        if self.loss_type == 'ce':
            val_loss = self.loss_fn(logits, label_idx)
        elif self.loss_type == 'regression':
            val_loss = F.mse_loss(logits, runtime)
        else: # Expected runtime loss
            probs = F.normalize(torch.sigmoid(logits), p=1)
            exp_runtime = torch.sum(torch.mul(probs, runtime), dim=1)
            val_loss = F.mse_loss(exp_runtime, min_runtime) / bs

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)

        # Return top1 and top2 runtime, min runtime and base runtime.
        pred_idx_top1 = torch.argmax(logits, dim=1)
        pred_time_top1 = runtime[torch.arange(bs), pred_idx_top1]

        _, pred_idx_top2 = torch.topk(logits, 2, dim=1)
        tmp = torch.zeros((bs, 2), device=labels.device)
        tmp[:, 0] = runtime[torch.arange(bs), pred_idx_top2[:, 0]]
        tmp[:, 1] = runtime[torch.arange(bs), pred_idx_top2[:, 1]]
        pred_time_top2, _ = tmp.min(dim=1)

        base_runtime = runtime[:, 0]
        return {'pred_time_top1': pred_time_top1, 'pred_time_top2': pred_time_top2, 
                'min_time': min_runtime, 'base_time': base_runtime, 'probs': logits, 'labels': labels}

    def validation_epoch_end(self, validation_step_outputs):
        keys = [k for k in validation_step_outputs[0].keys()]
        keys.remove('probs')  # Probs and labels will be saved separately for result analysis
        keys.remove('labels')
        all_val_outputs = {k: torch.concat([out[k] for out in validation_step_outputs]) for k in keys}
        hr_threshold = 300
        metrics = self.compute_metrics(all_val_outputs, hr_threshold)
        self.log_metrics(metrics, str(hr_threshold))
    
    def log_metrics(self, metrics, prefix):
        self.log(prefix+'_halt_rate_min', metrics['hr_min'], sync_dist=True)
        self.log(prefix+'_halt_rate_top1', metrics['hr_top1'], sync_dist=True)
        self.log(prefix+'_halt_rate_top2', metrics['hr_top2'], sync_dist=True)
        self.log(prefix+'_runtime_min', metrics['rt_min'], sync_dist=True)
        self.log(prefix+'_runtime_top1', metrics['rt_top1'], sync_dist=True)
        self.log(prefix+'_runtime_top2', metrics['rt_top2'], sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, test_step_outputs):
        # Compute the metrics based on all the test data
        keys = [k for k in test_step_outputs[0].keys()]
        keys.remove('probs')  # Probs and labels will be saved separately for result analysis
        keys.remove('labels')
        all_val_outputs = {k: torch.concat([out[k] for out in test_step_outputs]) for k in keys}
        test_metrics_list = []
        hr_options = [200, 250, 300, 350, 400, 450, 500]
        for hr_threshold in hr_options:
            metrics = self.compute_metrics(all_val_outputs, hr_threshold)
            self.log_metrics(metrics, str(hr_threshold))
            test_metrics_list.append(metrics)
        test_metrics = pd.DataFrame.from_records(test_metrics_list, index=hr_options).astype(float)
        save_path = os.path.join(self.logger.log_dir, 'test_metrics.csv')
        test_metrics.to_csv(save_path)
        print(f"Test metrics is saved to: {save_path}")

        # Save all predictions for result analysis
        test_probs = torch.concat([out['probs'] for out in test_step_outputs], dim=0).cpu().numpy()
        test_probs = pd.DataFrame(test_probs)
        test_probs_path = os.path.join(self.logger.log_dir, 'test_pred_probs.csv')
        test_probs.to_csv(test_probs_path, index=False)
        print(f"Prob prediction is saved to: {test_probs_path}")

        test_labels = torch.concat([out['labels'] for out in test_step_outputs], dim=0).cpu().numpy()
        test_labels = pd.DataFrame(test_labels)
        test_labels_path = os.path.join(self.logger.log_dir, 'test_labels.csv')
        test_labels.to_csv(test_labels_path, index=False)
        print(f"Test data label is saved to: {test_labels_path}")

    def configure_optimizers(self):
        if self.optim_type == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer

    def compute_metrics(self, outputs, hr_threshold):
        # Compute runtime (rt) and halt rate (hr) based on different halt rate thresholds.
        base_hr_cnt = (outputs['base_time'] > hr_threshold).sum().float()
        min_hr_cnt = (outputs['min_time'] > hr_threshold).sum().float()
        hr_min = 100. * (base_hr_cnt - min_hr_cnt) / base_hr_cnt 

        hr_cnt_top1 = (outputs['pred_time_top1'] > hr_threshold).sum().float()
        hr_top1 = 100. * (base_hr_cnt - hr_cnt_top1) / base_hr_cnt
        hr_cnt_top2 = (outputs['pred_time_top2'] > hr_threshold).sum().float()
        hr_top2 = 100. * (base_hr_cnt - hr_cnt_top2) / base_hr_cnt

        rt_top1 = outputs['pred_time_top1'].mean()
        rt_top2 = outputs['pred_time_top2'].mean()
        rt_min = outputs['min_time'].mean()
        rt_base = outputs['base_time'].mean()
        return {'hr_min': hr_min, 'hr_top1': hr_top1, 'hr_top2': hr_top2,
                'rt_min': rt_min, 'rt_top1': rt_top1, 'rt_top2': rt_top2, 'rt_base': rt_base}
    
    def log(self, metric_name, metric_val, **kwargs):
        self.logger.log_metrics((metric_name, metric_val), **kwargs)

# class AlternateTraining(BaseFinetuning):
#     def __init__(self, alternate_at_epoch=20):
#         super().__init__()
#         self.alternate_at_epoch = alternate_at_epoch
    
#     def freeze_before_training(self, pl_module):
#         self.freeze(pl_module.sat_encoder)

#     def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
#         if current_epoch == self.alternate_at_epoch:
#             self.freeze(pl_module.solver_encoder)
#             self.make_trainable(pl_module.sat_encoder)
#             print(f"INFO: Start training sat_encoder while freezing solver_encoder at epoch: {current_epoch}")
#             # self.unfreeze_and_add_param_group(modules=pl_module.sat_encoder, optimizer=optimizer, train_bn=True)

if __name__ == '__main__':
    pass
    # import sys
    # sys.path.append(os.getcwd())
    # from model.encoder.encoder import *
    # from model.decoder.decoder import *
    # solver_enc_mode = 'none'
    # sat_encoder = SATInstanceEncoder(10, 200)
    # solver_encoder = SolverEncoder(2304, 200, solver_enc_mode=solver_enc_mode)
    # cls_decoder = ClassifierDecoder(200, 200, num_classes=7, solver_enc_mode=solver_enc_mode)
    # module = SATSolverModule(sat_encoder, solver_encoder, cls_decoder, solver_enc_mode=solver_enc_mode)

    # import pickle as pkl
    # batch_0 = pkl.load(open('dataset/test_data/batch_0.pkl', 'rb'))
    # label_0 = pkl.load(open('dataset/test_data/label_0.pkl', 'rb'))
    # info_0 = pkl.load(open('dataset/test_data/info_0.pkl', 'rb'))

    # module.training_step((batch_0, label_0, info_0), 0)
    # module.validation_step((batch_0, label_0, info_0), 0)
    # print('foo')