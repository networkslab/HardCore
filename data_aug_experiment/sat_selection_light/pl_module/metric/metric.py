import torch
from torchmetrics import Metric

class HaltRate(Metric):
    def __init__(self, ot_threshold = 300):
        super().__init__()
        self.ot_threshold = ot_threshold
        self.add_state('pred_ot', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('base_ot', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, pred_idx: torch.Tensor, labels: torch.Tensor):
        runtime = labels[:, 2:-1]
        if len(pred_idx.shape) > 1:
            len_label = labels.shape[0]
            pred_time_all = torch.zeros((len_label, 2), device=pred_idx.device)
            pred_time_all[:, 0] = runtime[torch.arange(len_label), pred_idx[:, 0]]
            pred_time_all[:, 1] = runtime[torch.arange(len_label), pred_idx[:, 1]]
            pred_time, _ = pred_time_all.min(dim=1)
        else:
            if pred_idx[0] == -1:
                pred_time = labels[torch.arange(len(labels)), pred_idx]
            else:
                pred_time = runtime[torch.arange(len(runtime)), pred_idx]

        pred_ot_cnt = (pred_time > self.ot_threshold).sum()
        self.pred_ot += pred_ot_cnt

        base_ot_cnt = (labels[:, 2] > self.ot_threshold).sum()    # The runtime for Kissat3.0
        # min_ot_cnt = (labels[:, -1] > self.ot_threshold).sum()   # The min runtime for each SAT instance
        self.base_ot += base_ot_cnt
        # self.min_ot += min_ot_cnt

    def compute(self):
        ot_dec = 100. * (self.base_ot.float() - self.pred_ot.float()) / self.base_ot.float()
        return ot_dec

class AvgRuntime(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('avg_runtime', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('cnt', default=torch.tensor(0), dist_reduce_fx='sum')
    
    def update(self, pred_idx: torch.Tensor, labels: torch.Tensor):
        runtime = labels[:, 2:-1]
        if len(pred_idx.shape) > 1:     # Top 2 prediction
            len_label = labels.shape[0]
            pred_time_all = torch.zeros((len_label, 2), device=pred_idx.device)
            pred_time_all[:, 0] = runtime[torch.arange(len_label), pred_idx[:, 0]]
            pred_time_all[:, 1] = runtime[torch.arange(len_label), pred_idx[:, 1]]
            pred_time, _ = pred_time_all.min(dim=1)
        else:
            if pred_idx[0] == -1:
                pred_time = labels[torch.arange(len(labels)), pred_idx]
            else:
                pred_time = runtime[torch.arange(len(runtime)), pred_idx]
        self.avg_runtime += pred_time.float().mean()
        self.cnt += 1

    def compute(self):
        return self.avg_runtime / self.cnt