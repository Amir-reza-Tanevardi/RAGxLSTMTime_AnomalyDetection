from torch.utils.data import Dataset
import torch
import random

class TorchDataset(Dataset):
    """
    PyTorch Dataset for time-series windows + optional retrieval.
    Expects `dataset` to provide:
      - train_windows: np.ndarray [N_train x seq_len x D]
      - train_labels:  np.ndarray [N_train x ...]
      - val_windows:   np.ndarray [N_val x seq_len x D]
      - val_labels:    np.ndarray [N_val x ...]
    """

    def __init__(self, dataset, mode, seq_len, kwargs, device):
        super().__init__()
        self.mode = mode
        self.device = device
        self.seq_len = seq_len

        # raw data from your Dataset
        self.X_train = torch.from_numpy(dataset.train_windows).float()
        self.y_train = torch.from_numpy(dataset.train_labels)
        self.X_val   = torch.from_numpy(dataset.val_windows).float()
        self.y_val   = torch.from_numpy(dataset.val_labels)

        # retrieval settings
        self.use_retrieval  = (kwargs.exp_retrieval_type.lower() != 'none')
        self.num_candidates = kwargs.exp_retrieval_num_candidate_helpers

        # move whole dataset to GPU if desired
        batch_flag = (kwargs.exp_batch_size != -1) if mode=='train' else (kwargs.exp_val_batchsize != -1)
        if kwargs.full_dataset_cuda or not batch_flag:
            self.X_train = self.X_train.to(device)
            self.y_train = self.y_train.to(device)
            self.X_val   = self.X_val.to(device)
            self.y_val   = self.y_val.to(device)

        # sanity check
        if self.use_retrieval:
            assert self.num_candidates < len(self.X_train), (
                "exp_retrieval_num_candidate_helpers must be fewer than train windows"
            )

    def __len__(self):
        return self.X_train.size(0) if self.mode=='train' else self.X_val.size(0)

    def __getitem__(self, idx):
        # select window + label
        if self.mode == 'train':
            x = self.X_train[idx]        # [seq_len, D]
            y = self.y_train[idx]
        else:
            x = self.X_val[idx]
            y = self.y_val[idx]

        # retrieval candidates
        if self.use_retrieval and self.mode=='train':
            cand = self._sample_candidates(idx)
        else:
            cand = None

        # ensure on device
        x = x.to(self.device)
        if cand is not None:
            cand = cand.to(self.device)
        y = y.to(self.device)

        return x, cand, y

    def _sample_candidates(self, idx):
        """
        Randomly sample `num_candidates` windows from the training set,
        excluding the current index. Returns Tensor[K, seq_len, D].
        """
        N = self.X_train.size(0)
        choices = list(range(N))
        choices.remove(idx)
        sampled = random.sample(choices, self.num_candidates)
        return self.X_train[sampled]  # [K, seq_len, D]

    def set_mode(self, mode):
        """
        Switch between 'train' and 'val'.
        """
        assert mode in ('train','val')
        self.mode = mode
        # if full_dataset_cuda was toggled, you may re-send tensors to device here
