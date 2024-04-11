import os
import pandas as pd
from typing import Any, Dict, Mapping, Optional, Union

class Logger():
    def __init__(
        self,
        save_dir = "lightning_logs",
        name = None,
        version = None,
        **kwargs
    ):
        super().__init__()
        save_dir = os.fspath(save_dir)
        self._save_dir = save_dir
        self._name = name or ""
        self._version = version
        self.root_dir = os.path.join(self._save_dir, self._name)

        self.create_log_dir()
        self.step_metrics = {}
        self.epoch_metrics = {}
        self.epoch_buffer = {}

    def create_log_dir(self):
        if self._version is None:
            for i in range(100):
                if not os.path.isdir(os.path.join(self.root_dir, 'version_'+str(i))):
                    self._version = 'version_'+str(i)
                    break
        self.log_dir = os.path.join(self.root_dir, self._version)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def root_dir(self):
        return self.root_dir 

    def log_dir(self):
        return self.log_dir

    def log_metrics(self, metrics: Mapping[str, float], **kwargs):
        metric_name, metric_val = metrics
        # Store in a buffer within each epoch
        if metric_name in self.epoch_buffer.keys():
            self.epoch_buffer[metric_name].append(metric_val.item())
        else:
            self.epoch_buffer[metric_name] = [metric_val.item()]

        if metric_name.startswith('train'):
            if metric_name in self.step_metrics.keys():
                self.step_metrics[metric_name].append(metric_val.item())
            else:
                self.step_metrics[metric_name] = [metric_val.item()]
        
    def on_epoch_end(self):
        epoch_metric = {}
        for k, v in self.epoch_buffer.items():
            avg_val = sum(v) / len(v)
            if k in self.epoch_metrics.keys():
                self.epoch_metrics[k].append(avg_val)
            else:
                self.epoch_metrics[k] = [avg_val]
            epoch_metric[k] = avg_val
        print(epoch_metric)
        self.epoch_buffer = {}

    def save_metrics(self):
        step_metric_path = os.path.join(self.log_dir, 'step_metrics.csv')
        step_df = pd.DataFrame.from_dict(self.step_metrics)
        step_df.to_csv(step_metric_path)

        epoch_metric_path = os.path.join(self.log_dir, 'epoch_metrics.csv')
        epoch_df = pd.DataFrame.from_dict(self.epoch_metrics)
        epoch_df.to_csv(epoch_metric_path)


        