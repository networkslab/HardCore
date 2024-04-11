import os
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor

import dgl
import dgl.nn.pytorch as dglnn
import pandas as pd

# import pytorch_lightning as pl
from model.encoder.encoder import SATInstanceEncoder, SolverEncoder, SATzillaEncoder, SolverEncoderTFM
from model.decoder.decoder import ClassifierDecoder

from .utils.solver_graph_loader import load_solver_graphs, load_solver_prof_graphs, load_solver_emb
# from .metric.metric import HaltRate, AvgRuntime
# from pytorch_lightning.callbacks import BaseFinetuning

class TFMLayer(nn.Module):
    def __init__(self, q_dim, kv_dim, hidden_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads 
        self.ln_in = nn.LayerNorm(kv_dim, elementwise_affine=False)
        self.ln_hid = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads=self.num_heads, batch_first=True)
        self.ffn = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, q_in, kv_in, key_padding_mask):
        kv_norm = self.ln_in(kv_in)
        q = self.q_proj(q_in)
        k = self.k_proj(kv_norm)
        v = self.v_proj(kv_norm)
        attn_out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        # h2 = attn_out + kv_in
        # h_out = self.ffn(self.ln_hid(h2)) + h2
        return attn_out

class SATSolverCrossAttn(nn.Module):
    def __init__(self, feat_type, in_dim, hidden_dim, num_layers=2, solver_enc_mode='gnn', gnn_type='gcn', global_pool='mean', aux_task=None, aux_num_samples=10, manifold='lorentz'):
        super().__init__()
        self.num_heads = 2
        # self.ln_feat = nn.LayerNorm(in_dim, elementwise_affine=False)
        # self.mha_layers = nn.ModuleList()
        n_solver = 7
        q_dim = 200
        kv_dim = in_dim 
        self.ca_layer = TFMLayer(q_dim, kv_dim, hidden_dim, self.num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(n_solver*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_solver)
        )
        # for _ in range(num_layers-1):
        #     self.mha_layers.append(TFMLayer(hidden_dim, hidden_dim, self.num_heads))
        # Add a graph token at the beginning of each graph.
        # Implementation referred to https://github.com/microsoft/Graphormer/blob/77f436db46fb9013121289db670d1a763f264153/graphormer/modules/graphormer_layers.py#L103
        # self.graph_token = nn.Embedding(1, in_dim)
        # self.graph_token.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, sat_emb, solv_feats, key_padding_mask):
        bs = sat_emb.shape[0]
        n_solver = solv_feats.shape[0]

        # solv_feats_norm = self.ln_feat(solv_feats)

        out_list = []
        for i in range(n_solver):
            # solv_in = solv_feats_norm[i].unsqueeze(0).repeat((bs, 1, 1))    # dim=[bs, len, solv_emb_dim]
            solv_in = solv_feats[i].unsqueeze(0).repeat((bs, 1, 1))    # dim=[bs, len, solv_emb_dim]
            sat_in = sat_emb.unsqueeze(1)   # dim=[bs, 1, sat_emb_dim]
            kp_mask = key_padding_mask[i].unsqueeze(0).repeat((bs, 1))  # dim=[bs, len]
            attn_out = self.ca_layer(sat_in, solv_in, kp_mask) # dim=[bs, 1, solv_emb_dim]
            out_list.append(attn_out)
        tmp = torch.cat(out_list, dim=1).reshape((bs, -1))   # dim=[bs, n_solver, solv_emb_dim]
        logits = self.mlp(tmp)   # dim=[bs, n_solver]
        return logits

    def get_attn_feats_and_mask(self, g, h):
        bs = len(g.batch_num_nodes())
        seq_len = g.batch_num_nodes().max()
        feat_dim = h.shape[1]

        attn_feats = torch.zeros((bs, seq_len, feat_dim), device=h.device)
        # attn_mask = torch.ones((bs, seq_len+1, seq_len+1), device=h.device)   # Seq_len+1 for graph_token
        key_padding_mask = torch.ones((bs, seq_len), device=h.device)
        idx = 0
        for i, num_nodes in enumerate(g.batch_num_nodes()):
            # Scale the features [percentage_rt, self_rt, children_rt, called_cnt] 
            feat = h[idx:idx+num_nodes, :]
            feat_scaled = (feat - feat.min(dim=0)[0]) / (feat.max(dim=0)[0] - feat.min(dim=0)[0]) 
            attn_feats[i, :num_nodes, :] = feat_scaled 
            # attn_mask[i, :num_nodes+1, :num_nodes+1] = 0   # Graph token will be added at the beginning of each graph
            key_padding_mask[i, :num_nodes] = 0
            idx += num_nodes
        
        # attn_mask = attn_mask.repeat(self.num_heads, 1, 1)
        # attn_mask = attn_mask.bool().repeat(self.num_heads, 1, 1)   # Convert to bool, where node with True will not be attended
        # return attn_feats, attn_mask
        key_padding_mask = key_padding_mask.bool()
        return attn_feats, key_padding_mask

class SATSolverModule(nn.Module):
    # def __init__(self, sat_encoder: nn.Module, solver_encoder: nn.Module, cls_decoder: nn.Module, solver_enc_mode='none', aux_task=None, aux_loss_coef=1000, pca_mode='graph_7', lr=0.001):
    def __init__(self, cfg):
        super().__init__()
        sat_enc_init_args = cfg['sat_encoder']['init_args']
        sat_enc_init_args['feat_type'] = cfg['sat_feat_type']
        if cfg['sat_data'] == 'satzilla':
            self.sat_encoder = SATzillaEncoder(33)
        else:
            self.sat_encoder = SATInstanceEncoder(**sat_enc_init_args)


        # self.solver_bg = load_solver_graphs(pca_mode)              
        # solver_feats = self.solver_bg.ndata['feat']
        # self.solver_encoder = SolverEncoderTFM(**cfg['solver_encoder']['init_args'])
        # self.attn_feats, self.attn_mask = self.solver_encoder.get_attn_feats_and_mask(self.solver_bg, solver_feats)

        # self.cls_decoder = ClassifierDecoder(**cfg['cls_decoder']['init_args'])

        # self.num_classes = self.cls_decoder.num_classes
        
        self.sat_feat_type = cfg['sat_feat_type']
        self.solver_enc_mode = cfg['solver_enc_mode']
        self.aux_task = cfg['aux_task']
        self.aux_loss_coef = cfg['aux_loss_coef']
        pca_mode = cfg['pca_mode']

        if self.solver_enc_mode == 'gnn':
            # self.solver_bg = load_solver_graphs(pca_mode)              
            # solver_feats = self.solver_bg.ndata['feat']
            self.solver_bg = load_solver_prof_graphs()              
            solver_feats = self.solver_bg.ndata['prof_feat']
            self.cross_attn = SATSolverCrossAttn(**cfg['solver_encoder']['init_args'])
            # self.solver_encoder = SolverEncoderTFM(**cfg['solver_encoder']['init_args'])
            self.attn_feats, self.attn_mask = self.cross_attn.get_attn_feats_and_mask(self.solver_bg, solver_feats)
        elif self.solver_enc_mode == 'none':
            self.decoder = nn.Linear(200, 7)
        
        # if self.solver_enc_mode == 'gnn':
        #     # The solver graph is batched bi-directed graphs with LPE, RWPE, node_deg, feat (pca_50)
        #     self.solver_bg = load_solver_graphs(pca_mode)              
        #     # un_bg = dgl.unbatch(self.solver_bg_directed)
        #     # self.solver_bg = dgl.batch([dgl.to_bidirected(g) for g in un_bg])
        #     self.solver_feat_type = cfg['solver_encoder']['init_args']['feat_type']
        #     if self.solver_feat_type in ['lpe', 'rwpe']:
        #         solv_in_dim = 10
        #     elif self.solver_feat_type == 'node_deg':
        #         solv_in_dim = 4 
        #     elif self.solver_feat_type == 'codebert':
        #         solv_in_dim = 50 
        #     elif self.solver_feat_type == 'codebert_lpe':
        #         solv_in_dim = 60 
        #     elif self.solver_feat_type == 'codebert_lpe_deg':
        #         solv_in_dim = 64 
        #     cfg['solver_encoder']['init_args']['in_dim'] = solv_in_dim 
        #     self.solver_encoder = SolverEncoder(**cfg['solver_encoder']['init_args'])
        # elif self.solver_enc_mode == 'mha':
        #     self.solver_bg = load_solver_graphs(pca_mode)              
        #     solver_feats = self.solver_bg.ndata['feat']
        #     self.solver_encoder = SolverEncoderTFM(**cfg['solver_encoder']['init_args'])
        #     self.attn_feats, self.attn_mask = self.solver_encoder.get_attn_feats_and_mask(self.solver_bg, solver_feats)

        # elif self.solver_enc_mode == 'static':
        #     self.solver_static_emb = load_solver_emb(pca_mode) 
        # elif self.solver_enc_mode == 'lookup':
        #     # self.solver_lu_tbl = nn.Embedding(self.num_classes, 200)
        #     self.solver_lu_tbl = nn.Embedding(self.num_classes, 33)
        # elif self.solver_enc_mode == 'lookup_mlp':
        #     # self.solver_lu_tbl = nn.Embedding(self.num_classes, 200)
        #     self.solver_lu_tbl = nn.Embedding(self.num_classes, 33)
        #     self.solver_encoder = nn.Sequential(nn.Linear(33, 200),
        #                                          nn.ReLU(),
        #                                          nn.Linear(200, 33),
        #                                          nn.ReLU())
        # elif self.solver_enc_mode == 'static_lookup':
        #     self.solver_lu_tbl =  nn.Embedding(self.num_classes, 100)
        #     self.solver_static_emb = load_solver_emb(pca_mode)
        # elif self.solver_enc_mode == 'static_lin':
        #     self.solver_static_emb = load_solver_emb(pca_mode) 
        #     self.solver_encoder = SolverEncoder(**cfg['solver_encoder']['init_args'])

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

        if isinstance(sat_graph, torch.Tensor):   # If uses SATzilla features instead of SAT graphs
            sat_h = self.sat_encoder(sat_graph)
        else:
            if self.sat_feat_type == 'pe':
                sat_feat = sat_graph.ndata['pe']
            else:
                sat_feat = sat_graph.ndata['node_type']

            sat_h = self.sat_encoder(sat_graph, sat_feat)


        if self.solver_enc_mode == 'gnn':
            self.attn_feats = self.attn_feats.to(self.device)
            self.attn_mask = self.attn_mask.to(self.device)
            logits = self.cross_attn(sat_h, self.attn_feats, self.attn_mask)
        elif self.solver_enc_mode == 'none':
            logits = self.decoder(sat_h)

        # if self.solver_enc_mode == 'gnn':
        #     # self.solver_bg_directed = self.solver_bg_directed.to(self.device)
        #     self.solver_bg = self.solver_bg.to(self.device)
        #     # solver_feat = self.solver_bg_directed.ndata['feat']
        #     if self.solver_feat_type == 'lpe':
        #         solver_feat = self.solver_bg.ndata['LPE']
        #     elif self.solver_feat_type == 'rwpe':
        #         solver_feat = self.solver_bg.ndata['RWPE']
        #     elif self.solver_feat_type == 'node_deg':
        #         solver_feat = self.solver_bg.ndata['node_deg']
        #     elif self.solver_feat_type == 'codebert':
        #         solver_feat = self.solver_bg.ndata['feat']
        #     elif self.solver_feat_type == 'codebert_lpe':
        #         solver_feat = torch.cat([self.solver_bg.ndata['feat'], 
        #                                  self.solver_bg.ndata['LPE']], dim=1) 
        #     elif self.solver_feat_type == 'codebert_lpe_deg':
        #         solver_feat = torch.cat([self.solver_bg.ndata['feat'], 
        #                                  self.solver_bg.ndata['LPE'],
        #                                  self.solver_bg.ndata['node_deg']], dim=1) 
        #     else:
        #         solver_feat = self.solver_bg.ndata['feat']
        #     solver_h = self.solver_encoder(self.solver_bg, solver_feat)
        #     if self.aux_task:
        #         aux_pred, aux_label = self.solver_encoder.aux_task_forward(self.solver_bg_directed)
        #         self.aux_loss = self.aux_loss_func(aux_pred, aux_label)
        # elif self.solver_enc_mode == 'mha':
        #     self.attn_feats = self.attn_feats.to(self.device)
        #     self.attn_mask = self.attn_mask.to(self.device)
        #     solver_h = self.solver_encoder(self.attn_feats, self.attn_mask)
        # elif self.solver_enc_mode == 'static':
        #     solver_h = self.solver_static_emb.to(self.device)
        # elif self.solver_enc_mode == 'lookup':
        #     solver_h = self.solver_lu_tbl(torch.arange(self.num_classes, device=self.device))
        # elif self.solver_enc_mode == 'lookup_mlp':
        #     solver_emb = self.solver_lu_tbl(torch.arange(self.num_classes, device=self.device))
        #     solver_h = self.solver_encoder(solver_emb)
        # elif self.solver_enc_mode == 'static_lookup':
        #     solver_static_h = self.solver_static_emb.to(self.device)
        #     solver_lookup_h = self.solver_lu_tbl(torch.arange(self.num_classes, device=self.device))
        #     solver_h = torch.concat([solver_static_h, solver_lookup_h], dim=1)
        # elif self.solver_enc_mode == 'static_lin':
        #     solver_h = self.solver_static_emb.to(self.device)
        #     solver_h = self.solver_encoder(None, solver_h)
        # else:
        #     solver_h = None
        # logits = self.cls_decoder(sat_h, solver_h)
        return logits

    def training_step(self, batch, batch_idx):
        # labels in dataset includes the following info: ['#var', '#clause', 'base', 'HyWalk', 'MOSS', 'ESA', 'bulky', 'UCB', 'MIN'] 
        # Note that g is batched graphs
        sat_graph, labels, graph_info = batch
        if labels.shape[1] > 7:
            runtime = labels[:, 2:-1]   # [#var, #clause, runtime_solvers, min_runtime]
        else:
            runtime = labels
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
        min_runtime, label_idx = runtime.min(dim=1)

        bs = len(labels) 

        # Call forward step
        logits = self.forward_step(batch)

        # Cross Entropy loss or Expected runtime loss
        if self.loss_type == 'ce':
            val_loss = self.loss_fn(logits, label_idx)
        else: # Expected runtime loss
            logits = self.forward_step(batch)
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