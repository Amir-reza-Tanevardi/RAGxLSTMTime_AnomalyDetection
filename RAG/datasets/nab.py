# datasets/nab.py

import os
import json
import numpy as np
import pandas as pd
from datasets.base import BaseDataset

class NABDataset(BaseDataset):
    """
    Numenta Anomaly Benchmark loader with external JSON labels.
    Expects:
      - A CSV file `<data_set>.csv` in `data_path`,
          with at least columns ['timestamp', 'value'].
      - A JSON file at `anomaly_map_path` mapping
        "subdir/filename.csv" → [list of timestamp strings].
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('data_set')
        self.filename = f"{self.name}.csv"
        self.anomaly_map_path = kwargs.get('anomaly_map_path')
        if self.anomaly_map_path is None:
            raise ValueError("Must pass anomaly_map_path in args")
        
        # load the JSON once
        with open(self.anomaly_map_path, 'r') as f:
            self.anomaly_map = json.load(f)

    def load(self):
        csv_path = os.path.join(self.data_path, self.filename)
        df = pd.read_csv(csv_path)

        # get timestamps as strings, and float values
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            raise ValueError("CSV must have 'timestamp' and 'value' columns")
        timestamps = df['timestamp'].astype(str).tolist()
        X = df['value'].astype(float).values.reshape(-1, 1)

        # look up anomalies for this file key
        # JSON keys are relative paths, e.g. "artificialWithAnomaly/art_daily_jumpsup.csv"
        key = os.path.join(os.path.basename(os.path.dirname(self.filename)),
                           os.path.basename(self.filename))
        anomalies = set(self.anomaly_map.get(key, []))

        # build y: 1 if timestamp exactly in anomalies, else 0
        y = np.array([1 if ts in anomalies else 0 for ts in timestamps], dtype=int)

        self.raw_X = X
        self.raw_y = y
        self.T, self.D = X.shape

        print(f"Loaded NAB '{self.name}': T={self.T}, D={self.D}, "
              f"{y.sum()} anomalies")

    def __repr__(self):
        return (f"NABDataset('{self.name}'): "
                f"{self.T} timesteps, {self.D} features, "
                f"{int(self.raw_y.sum())} anomalies")
