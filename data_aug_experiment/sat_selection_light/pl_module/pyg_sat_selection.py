import os
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor

# import dgl
# import dgl.nn.pytorch as dglnn
import pandas as pd

# import pytorch_lightning as pl
# from model.encoder.encoder import SATInstanceEncoder, SATInstanceEncoderHetero, SATInstanceEncoderHeteroSAT, SolverEncoder, SATzillaEncoder, SolverEncoderTFM
from model.encoder.pyg_encoder import SATInstanceEncoderHeteroSAT
from model.decoder.decoder import ClassifierDecoder

from .utils.solver_graph_loader import load_solver_graphs, load_solver_emb
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
        elif cfg['sat_graph_type'] == 'hetero':
            self.sat_encoder = SATInstanceEncoderHetero(**sat_enc_init_args)
        elif cfg['sat_graph_type'] == 'hetero_satzilla':
            self.sat_encoder = SATInstanceEncoderHeteroSAT(**sat_enc_init_args)
        else:
            self.sat_encoder = SATInstanceEncoder(**sat_enc_init_args)
        self.cls_decoder = ClassifierDecoder(**cfg['cls_decoder']['init_args'])

        self.num_classes = self.cls_decoder.num_classes
        
        self.sat_feat_type = cfg['sat_feat_type']
        self.solver_enc_mode = cfg['solver_enc_mode']
        self.aux_task = cfg['aux_task']
        self.aux_loss_coef = cfg['aux_loss_coef']
        self.optim_type = cfg['optim_type']
        pca_mode = cfg['pca_mode']
        if self.solver_enc_mode == 'gnn':
            # The solver graph is batched bi-directed graphs with LPE, RWPE, node_deg, feat (pca_50)
            self.solver_bg = load_solver_graphs(pca_mode)              
            # un_bg = dgl.unbatch(self.solver_bg_directed)
            # self.solver_bg = dgl.batch([dgl.to_bidirected(g) for g in un_bg])
            self.solver_feat_type = cfg['solver_encoder']['init_args']['feat_type']
            if self.solver_feat_type in ['lpe', 'rwpe']:
                solv_in_dim = 10
            elif self.solver_feat_type == 'node_deg':
                solv_in_dim = 4 
            elif self.solver_feat_type == 'codebert':
                solv_in_dim = 50 
            elif self.solver_feat_type == 'codebert_lpe':
                solv_in_dim = 60 
            elif self.solver_feat_type == 'codebert_lpe_deg':
                solv_in_dim = 64 
            cfg['solver_encoder']['init_args']['in_dim'] = solv_in_dim 
            self.solver_encoder = SolverEncoder(**cfg['solver_encoder']['init_args'])
        elif self.solver_enc_mode == 'mha':
            self.solver_bg = load_solver_graphs(pca_mode)              
            solver_feats = self.solver_bg.ndata['feat']
            self.solver_encoder = SolverEncoderTFM(**cfg['solver_encoder']['init_args'])
            self.attn_feats, self.attn_mask = self.solver_encoder.get_attn_feats_and_mask(self.solver_bg, solver_feats)

        elif self.solver_enc_mode == 'static':
            self.solver_static_emb = load_solver_emb(pca_mode) 
        elif self.solver_enc_mode == 'lookup':
            # self.solver_lu_tbl = nn.Embedding(self.num_classes, 200)
            self.solver_lu_tbl = nn.Embedding(self.num_classes, 33)
        elif self.solver_enc_mode == 'lookup_mlp':
            # self.solver_lu_tbl = nn.Embedding(self.num_classes, 200)
            self.solver_lu_tbl = nn.Embedding(self.num_classes, 33)
            self.solver_encoder = nn.Sequential(nn.Linear(33, 200),
                                                 nn.ReLU(),
                                                 nn.Linear(200, 33),
                                                 nn.ReLU())
        elif self.solver_enc_mode == 'static_lookup':
            self.solver_lu_tbl =  nn.Embedding(self.num_classes, 100)
            self.solver_static_emb = load_solver_emb(pca_mode)
        elif self.solver_enc_mode == 'static_lin':
            self.solver_static_emb = load_solver_emb(pca_mode) 
            self.solver_encoder = SolverEncoder(**cfg['solver_encoder']['init_args'])

        if self.aux_task:
            self.aux_loss_func = nn.BCELoss()
        self.lr = float(cfg['lr'])

        self.loss_type = cfg['loss_type']
        if self.loss_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward_step(self, batch):
        sat_graph, labels, graph_info = batch

        # if isinstance(sat_graph, torch.Tensor):   # If uses SATzilla features instead of SAT graphs
        #     sat_h = self.sat_encoder(sat_graph)
        # else:
        #     if self.sat_feat_type == 'pe':
        #         sat_feat = sat_graph.ndata['pe']
        #     elif self.sat_feat_type == 'satzilla':
        #         sat_feat = sat_graph.ndata['satzilla']
        #     else:
        #         sat_feat = sat_graph.ndata['node_type']
            
        batch_num_nodes = []
        batch_num_nodes.append(sat_graph.batch_num_nodes('pos_lit'))
        batch_num_nodes.append(sat_graph.batch_num_nodes('neg_lit'))
        batch_num_nodes.append(sat_graph.batch_num_nodes('clause'))
        
        sat_feat_list = []
        sat_feat_list.append(sat_graph.nodes['pos_lit'].data['satzilla'])
        sat_feat_list.append(sat_graph.nodes['neg_lit'].data['satzilla'])
        sat_feat_list.append(sat_graph.nodes['clause'].data['satzilla'])

        edge_index_list = []
        edge_index_list.append(torch.stack(sat_graph[('pos_lit', 'in', 'clause')].edges(), dim=0))
        edge_index_list.append(torch.stack(sat_graph[('neg_lit', 'in', 'clause')].edges(), dim=0))
        edge_index_list.append(torch.stack(sat_graph[('clause', 'contain', 'pos_lit')].edges(), dim=0))
        edge_index_list.append(torch.stack(sat_graph[('clause', 'contain', 'neg_lit')].edges(), dim=0))
        edge_index_list.append(torch.stack(sat_graph[('pos_lit', 'flip', 'neg_lit')].edges(), dim=0))
        edge_index_list.append(torch.stack(sat_graph[('neg_lit', 'flip', 'pos_lit')].edges(), dim=0))

        sat_h = self.sat_encoder(batch_num_nodes, sat_feat_list, edge_index_list)

        if self.solver_enc_mode == 'gnn':
            # self.solver_bg_directed = self.solver_bg_directed.to(self.device)
            self.solver_bg = self.solver_bg.to(self.device)
            # solver_feat = self.solver_bg_directed.ndata['feat']
            if self.solver_feat_type == 'lpe':
                solver_feat = self.solver_bg.ndata['LPE']
            elif self.solver_feat_type == 'rwpe':
                solver_feat = self.solver_bg.ndata['RWPE']
            elif self.solver_feat_type == 'node_deg':
                solver_feat = self.solver_bg.ndata['node_deg']
            elif self.solver_feat_type == 'codebert':
                solver_feat = self.solver_bg.ndata['feat']
            elif self.solver_feat_type == 'codebert_lpe':
                solver_feat = torch.cat([self.solver_bg.ndata['feat'], 
                                         self.solver_bg.ndata['LPE']], dim=1) 
            elif self.solver_feat_type == 'codebert_lpe_deg':
                solver_feat = torch.cat([self.solver_bg.ndata['feat'], 
                                         self.solver_bg.ndata['LPE'],
                                         self.solver_bg.ndata['node_deg']], dim=1) 
            else:
                solver_feat = self.solver_bg.ndata['feat']
            solver_h = self.solver_encoder(self.solver_bg, solver_feat)
            if self.aux_task:
                aux_pred, aux_label = self.solver_encoder.aux_task_forward(self.solver_bg_directed)
                self.aux_loss = self.aux_loss_func(aux_pred, aux_label)
        elif self.solver_enc_mode == 'mha':
            self.attn_feats = self.attn_feats.to(self.device)
            self.attn_mask = self.attn_mask.to(self.device)
            solver_h = self.solver_encoder(self.attn_feats, self.attn_mask)
        elif self.solver_enc_mode == 'static':
            solver_h = self.solver_static_emb.to(self.device)
        elif self.solver_enc_mode == 'lookup':
            solver_h = self.solver_lu_tbl(torch.arange(self.num_classes, device=self.device))
        elif self.solver_enc_mode == 'lookup_mlp':
            solver_emb = self.solver_lu_tbl(torch.arange(self.num_classes, device=self.device))
            solver_h = self.solver_encoder(solver_emb)
        elif self.solver_enc_mode == 'static_lookup':
            solver_static_h = self.solver_static_emb.to(self.device)
            solver_lookup_h = self.solver_lu_tbl(torch.arange(self.num_classes, device=self.device))
            solver_h = torch.concat([solver_static_h, solver_lookup_h], dim=1)
        elif self.solver_enc_mode == 'static_lin':
            solver_h = self.solver_static_emb.to(self.device)
            solver_h = self.solver_encoder(None, solver_h)
        else:
            solver_h = None
        logits = self.cls_decoder(sat_h, solver_h)
        return logits

    def training_step(self, batch, batch_idx):
        # labels in dataset includes the following info: ['#var', '#clause', 'base', 'HyWalk', 'MOSS', 'mabgb', 'ESA', 'bulky', 'UCB', 'MIN'] 
        # Note that g is batched graphs
        sat_graph, labels, graph_info = batch
        if labels.shape[1] > 7:
            runtime = labels[:, 2:-1]   # [#var, #clause, runtime_solvers, min_runtime]
        else:
            runtime = labels
        
        # Remove ESA
        if self.num_classes == 6:
            runtime = torch.concat([runtime[:, :4], runtime[:, 5:]], dim=-1)
        elif self.num_classes == 5:   # Remove kissat3 and bulky
            runtime = torch.concat([runtime[:, 1:5], runtime[:, 6:]], dim=-1)

        min_runtime, label_idx = runtime.min(dim=1)

        bs = len(labels) 

        # Call forward step
        logits = self.forward_step(batch)

        # Cross Entropy loss or Expected runtime loss
        if self.loss_type == 'ce':
            train_loss = self.loss_fn(logits, label_idx)
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
        if labels.shape[1] > 7:
            runtime = labels[:, 2:-1]   # [#var, #clause, runtime_solvers, min_runtime]
        else:
            runtime = labels

        # Remove ESA
        if self.num_classes == 6:
            runtime = torch.concat([runtime[:, :4], runtime[:, 5:]], dim=-1)
        elif self.num_classes == 5:
            runtime = torch.concat([runtime[:, 1:5], runtime[:, 6:]], dim=-1)

        min_runtime, label_idx = runtime.min(dim=1)

        bs = len(labels) 

        logits = self.forward_step(batch)

        # Cross Entropy loss or Expected runtime loss
        if self.loss_type == 'ce':
            val_loss = self.loss_fn(logits, label_idx)
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

    def infer_step(self, batch, batch_idx):
        # There is only graph for inference. No label
        # sat_graph = batch
        cnf_filename = list(batch[2])
        logits = self.forward_step(batch)
        probs = F.softmax(logits, dim=-1)
        return {'cnf_filename': cnf_filename, 'probs': probs}
    
    def infer_epoch_end(self, infer_step_outputs):
        cnf_filenames = sum([out['cnf_filename'] for out in infer_step_outputs], [])
        infer_probs = torch.concat([out['probs'] for out in infer_step_outputs], dim=0).cpu().numpy()

        if self.num_classes == 6:
            infer_res = pd.DataFrame(infer_probs, columns=['kissat-3', 'HyWalk', 'MOSS', 'mabgb', 'bulky', 'UCB'], index=cnf_filenames)
        else:
            infer_res = pd.DataFrame(infer_probs, columns=['kissat-3', 'HyWalk', 'MOSS', 'mabgb', 'ESA', 'bulky', 'UCB'], index=cnf_filenames)
        infer_res_path = os.path.join(self.logger.log_dir, 'infer_res.csv')
        infer_res.to_csv(infer_res_path)
        print(f"Inference results are saved to: {infer_res_path}")

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