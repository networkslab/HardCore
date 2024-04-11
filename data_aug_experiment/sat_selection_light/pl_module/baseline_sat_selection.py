import os
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor

# import dgl
# import dgl.nn.pytorch as dglnn
import pandas as pd

class MLPClassifier(nn.Module):
    def __init__(self, in_dim=33, hidden_dim=200, num_layers=2, num_classes=7, **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.num_hid_layers = num_layers
        if num_layers > 0:
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            for i in range(num_layers-1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers = nn.Sequential(*layers)
            self.classify = nn.Linear(hidden_dim, num_classes)
        else:
            self.classify = nn.Linear(in_dim, num_classes)

    def forward(self, h):
        h = self.norm(h)
        if self.num_hid_layers > 0:
            for layer in self.layers:
                h = F.relu(layer(h))
        out = self.classify(h)   # The output is logits and need to be normalized using softmax
        return out
class MLPRegressor(nn.Module):
    def __init__(self, in_dim=33, hidden_dim=200, num_layers=2, num_classes=7, **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.num_hid_layers = num_layers
        if num_layers > 0:
            layers = []
            layers.append(nn.Linear(in_dim, hidden_dim))
            for i in range(num_layers-1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers = nn.Sequential(*layers)
            self.regress = nn.Linear(hidden_dim, num_classes)
        else:
            self.regress = nn.Linear(in_dim, num_classes)

    def forward(self, h):
        # print('mlp regressor')
        h = self.norm(h)
        if self.num_hid_layers > 0:
            for layer in self.layers:
                h = F.relu(layer(h))
        out = self.regress(h)   # The output is logits and need to be normalized using softmax
        return out
    
class SATzillaRegressionModule(nn.Module):
    # def __init__(self, sat_encoder: nn.Module, solver_encoder: nn.Module, cls_decoder: nn.Module, solver_enc_mode='none', aux_task=None, aux_loss_coef=1000, pca_mode='graph_7', lr=0.001):
    def __init__(self, cfg):
        super().__init__()
        self.model = MLPRegressor(**cfg['sat_encoder']['init_args'])
        # self.loss_fn = nn.CrossEntropyLoss()
        self.lr = float(cfg['lr'])
        self.cfg = cfg
        self.cfg['loss_type'] = 'MSE'

    def set_loss_fn(self, loss_weights=None):
        self.loss_type = self.cfg['loss_type']
        if self.loss_type == 'ce':
            if loss_weights != None:
                self.loss_fn = nn.MSELoss()
            else:
                self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        feats, labels, _ = batch
        if labels.shape[1] == 10:
            runtime = labels[:, 2:-1]
        else:
            runtime = labels
        probs = self.model(feats)
        # label_idx = runtime.argmin(dim=1)
        train_loss = self.loss_fn(probs, runtime)
        self.log('train_loss', train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        feats, labels, _ = batch
        if labels.shape[1] == 10:
            runtime = labels[:, 2:-1]
        else:
            runtime = labels
        probs = self.model(feats)
        # label_idx = runtime.argmin(dim=1)
        val_loss = self.loss_fn(probs, runtime)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)

        bs = feats.shape[0]
        device = runtime.device
        # Return top1 and top2 runtime, min runtime and base runtime.
        pred_idx_top1 = torch.argmin(probs, dim=1)
        pred_time_top1 = runtime[torch.arange(bs), pred_idx_top1]

        _, pred_idx_top2 = torch.topk(probs, 2, dim=1)
        tmp = torch.zeros((bs, 2), device=device)
        tmp[:, 0] = runtime[torch.arange(bs), pred_idx_top2[:, 0]]
        tmp[:, 1] = runtime[torch.arange(bs), pred_idx_top2[:, 1]]
        pred_time_top2, _ = tmp.min(dim=1)

        min_time, _ = runtime.min(dim=1)
        # base_idx = torch.tensor([2]*len(probs), dtype=torch.long, device=device)
        base_time = runtime[torch.arange(bs), 0]
        
        return {'pred_time_top1': pred_time_top1, 'pred_time_top2': pred_time_top2, 
                'min_time': min_time, 'base_time': base_time, 'probs': probs, 'labels': runtime}

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
        hr_options = [100, 200, 300, 400, 500]
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
        # try:
        #     for row in test_probs:
        #         a = [float(item) for item in row]
        # except:
        #     print(row)
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


class SATzillaModule(nn.Module):
    # def __init__(self, sat_encoder: nn.Module, solver_encoder: nn.Module, cls_decoder: nn.Module, solver_enc_mode='none', aux_task=None, aux_loss_coef=1000, pca_mode='graph_7', lr=0.001):
    def __init__(self, cfg):
        super().__init__()
        self.model = MLPClassifier(**cfg['sat_encoder']['init_args'])
        # self.loss_fn = nn.CrossEntropyLoss()
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

    def training_step(self, batch, batch_idx):
        feats, labels, _ = batch
        if labels.shape[1] == 10:
            runtime = labels[:, 2:-1]
        else:
            runtime = labels
        probs = self.model(feats)
        label_idx = runtime.argmin(dim=1)
        train_loss = self.loss_fn(probs, label_idx)
        self.log('train_loss', train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        feats, labels, _ = batch
        if labels.shape[1] == 10:
            runtime = labels[:, 2:-1]
        else:
            runtime = labels
        probs = self.model(feats)
        label_idx = runtime.argmin(dim=1)
        val_loss = self.loss_fn(probs, label_idx)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True)

        bs = feats.shape[0]
        device = runtime.device
        # Return top1 and top2 runtime, min runtime and base runtime.
        pred_idx_top1 = torch.argmax(probs, dim=1)
        pred_time_top1 = runtime[torch.arange(bs), pred_idx_top1]

        _, pred_idx_top2 = torch.topk(probs, 2, dim=1)
        tmp = torch.zeros((bs, 2), device=device)
        tmp[:, 0] = runtime[torch.arange(bs), pred_idx_top2[:, 0]]
        tmp[:, 1] = runtime[torch.arange(bs), pred_idx_top2[:, 1]]
        pred_time_top2, _ = tmp.min(dim=1)

        min_time, _ = runtime.min(dim=1)
        # base_idx = torch.tensor([2]*len(probs), dtype=torch.long, device=device)
        base_time = runtime[torch.arange(bs), 0]
        return {'pred_time_top1': pred_time_top1, 'pred_time_top2': pred_time_top2, 
                'min_time': min_time, 'base_time': base_time, 'probs': probs, 'labels': runtime}

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
        hr_options = [100, 200, 300, 400, 500]
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