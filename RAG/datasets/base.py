
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# datasets/base.py

import numpy as np

class BaseDataset:
    """
    Base class for time-series datasets with sliding-window support.
    Subclasses should override `load()` to populate:
      - self.raw_X : np.ndarray (T, D)
      - self.raw_y : np.ndarray (T,)
    After `load()`, you can call:
      - load_time_series(seq_len)
      - split_time_series_train_val(train_ratio, seed, iteration)
    """
    def __init__(self, **kwargs):
        self.data_path = kwargs.get('data_path', '.')
        self.name     = kwargs.get('data_set', 'dataset')
        self.raw_X = None
        self.raw_y = None
        self.T = None
        self.D = None

    def load(self):
        raise NotImplementedError("Subclasses must implement load()")

    def load_time_series(self, seq_len: int):
        assert self.raw_X is not None, "Call load() first"
        X, y = self.raw_X, self.raw_y
        T, D = X.shape
        self.T, self.D = T, D

        n_windows = T - seq_len + 1
        windows = np.zeros((n_windows, seq_len, D), dtype=X.dtype)
        labels  = np.zeros((n_windows,),      dtype=y.dtype)

        for i in range(n_windows):
            windows[i] = X[i:i+seq_len]
            labels[i]  = int(y[i:i+seq_len].any())

        self._all_windows = windows
        self._all_labels  = labels

        self.num_features  = list(range(self.D))
        self.cardinalities = []

    def split_time_series_train_val(self,
                                    train_ratio: float,
                                    seed: int,
                                    iteration: int):
        assert hasattr(self, '_all_windows'), "Call load_time_series() first"
        n = len(self._all_windows)
        split = int(train_ratio * n)
        self.train_windows = self._all_windows[:split]
        self.train_labels  = self._all_labels[:split]
        self.val_windows   = self._all_windows[split:]
        self.val_labels    = self._all_labels[split:]
        print(f"  → {len(self.train_windows)} train windows, "
              f"{len(self.val_windows)} val windows")
