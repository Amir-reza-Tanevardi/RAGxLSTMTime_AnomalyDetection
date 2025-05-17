import warnings
import torch
from torch.utils.data import DataLoader, Dataset

class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int = 0,
        collate_fn=None,
        shuffle_train: bool = False,
        shuffle_val: bool = False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        # Remove 'split' from kwargs to avoid overwrite
        self.dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k != "split"}

        # Prepare datasets and loaders
        self.train_dataset, self.train = self._make_dloader_with_dataset("train", shuffle=self.shuffle_train)
        self.valid_dataset, self.valid = self._make_dloader_with_dataset("val", shuffle=self.shuffle_val)
        self.test_dataset, self.test = self._make_dloader_with_dataset("test", shuffle=False)

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.valid

    def test_dataloader(self):
        return self.test

    def _make_dloader_with_dataset(self, split: str, shuffle: bool = False):
        assert split in {"train", "val", "test"}, f"Invalid split '{split}'"
        dataset = self.datasetCls(split=split, **self.dataset_kwargs)

        if len(dataset) == 0:
            warnings.warn(f"No data found for split: {split}")
            return None, None

        print(f"[{split.upper()}] Dataset loaded: {len(dataset)} samples")

        dataloader = DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
        )
        return dataset, dataloader

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for PyTorch DataLoader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        """
        Wraps raw Dataset or list of samples into a DataLoader.
        """
        from torch.utils.data import DataLoader

        if isinstance(test_data, DataLoader):
            return test_data

        if not isinstance(test_data, Dataset):
            raise ValueError("test_data must be a torch Dataset or DataLoader.")

        if batch_size is None:
            batch_size = self.batch_size

        return DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def get_labels(self, split: str):
        """
        Access the labels for a given split.
        Args:
            split (str): One of 'train', 'val', or 'test'.
        Returns:
            labels (np.ndarray or list) or None
        """
        if split == "train":
            return getattr(self.train_dataset, "labels", None)
        elif split == "val":
            return getattr(self.valid_dataset, "labels", None)
        elif split == "test":
            return getattr(self.test_dataset, "labels", None)
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")
