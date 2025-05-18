import os, glob, re, pickle, sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from optim import LRScheduler
from torch_dataset import TorchDataset, collate_with_optional_cand
from utils.log_utils import make_job_name
from utils.checkpoint_utils import EarlyStopCounter, EarlyStopSignal
from loss import Loss
from mask import apply_mask, generate_mask_val
from sklearn.metrics import precision_recall_fscore_support as prf
from torchmetrics import AveragePrecision, AUROC


class Trainer():
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss: Loss,
        scaler,
        kwargs,
        torch_dataset: TorchDataset,
        distributed_args: dict,
        device: torch.device,
    ):
        # ------ unchanged boilerplate ------
        self.kwargs = kwargs
        self.device = device
        self.iter_seed = kwargs.iter_seed
        self.n_epochs = kwargs.exp_train_total_epochs
        self.p_mask_train = kwargs.model_augmentation_bert_mask_prob['train']
        self.p_mask_val   = kwargs.model_augmentation_bert_mask_prob['val']

        self.is_distributed = distributed_args is not None
        if self.is_distributed:
            self.world_size = distributed_args['world_size']
            self.rank       = distributed_args['rank']
            self.gpu        = distributed_args['gpu']

        self.model     = model
        self.optimizer = optimizer
        self.scaler    = scaler
        self.scheduler = LRScheduler(c=kwargs,
                                     name=kwargs.exp_scheduler,
                                     optimizer=optimizer)
        self.loss      = loss
        self.epoch = 0

        # our new dataset, returns (x, cand, y)
        self.dataset = torch_dataset

        # batch sizes
        self.is_batch_learning = (kwargs.exp_batch_size != -1)
        self.is_batch_eval     = (kwargs.exp_val_batchsize != -1)
        self.batch_size        = (kwargs.exp_batch_size 
                                  if self.is_batch_learning 
                                  else len(self.dataset))
        self.batch_size_val    = (kwargs.exp_val_batchsize
                                  if self.is_batch_eval
                                  else len(self.dataset))

        # generate val masks as before
        self.val_masks = generate_mask_val(
            data_shape=(self.batch_size_val,          # B
                        self.dataset.seq_len,         # L
                        self.dataset.X_train.size(-1) # D
                      ),
            num_masks=1,
            p_mask=self.p_mask_val,
            deterministic=self.kwargs.exp_deterministic_masks_val
        )

        self.is_main_process = (not self.is_distributed) or (self.rank == 0)
        if self.is_main_process:
            self.job_name = make_job_name(kwargs, kwargs.np_seed, kwargs.iteration)
            self.res_dir  = os.path.join('results', self.dataset.dataset_name, self.job_name)
            os.makedirs(self.res_dir, exist_ok=True)
            self.checkpoint_dir = os.path.join('checkpoints', self.dataset.dataset_name, self.job_name)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.early_stop_counter = EarlyStopCounter(
            kwargs, self.job_name, self.dataset.dataset_name,
            device=self.device, is_distributed=self.is_distributed
        )

    def train(self):
        # switch to training mode
        self.dataset.set_mode('train')
        self.loss.set_mode('train')
        if not self.is_distributed:
            dl = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=0,
                            collate_fn=collate_with_optional_cand,
                            drop_last=True)
        else:
            dl = self.get_distributed_dataloader(self.batch_size, self.dataset)

        self.model.to(self.device)
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch
            epoch_start = datetime.now()
            if self.is_main_process:
                print(f"\n{' Training epoch '+str(epoch)+' ':#^80}")

            if self.is_distributed:
                dl.sampler.set_epoch(epoch)

            self.model.train()
            # set retrieval submodule
            if self.is_distributed:
                self.model.module.set_retrieval_module_mode('train')
            else:
                self.model.set_retrieval_module_mode('train')

            for x, cand, y in dl:
                # x: [B, L, D], cand: [B, K, L, D] or None, y: [...]
                # apply masking
                data = apply_mask(
                    data = x, 
                    p_mask=self.p_mask_train,
                    force_mask=self.kwargs.exp_force_all_masked,
                    device=self.device
                )

                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda',enabled=self.kwargs.model_amp):
                    if cand is not None:
                        out = self.model(data['masked_tensor'], cand)
                    else:
                        out = self.model(data['masked_tensor'])
                    # loss against ground truth y
                    self.loss.compute(
                        output=out,
                        ground_truth=data['ground_truth'],
                        mask_matrix=data['mask_matrix']
                    )
                    loss_dict = self.loss.finalize_batch_loss()
                    train_loss = loss_dict['train']['total_loss']
                    #self.loss.update_losses()
                    self.scaler.scale(train_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()

            # end-of-epoch bookkeeping
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            if self.is_main_process:
                print(f"Epoch {epoch} done in {epoch_time:.2f}s; loss = {train_loss.item():.4f}")

            # early stopping & periodic validation
            stop_signal, *_ = self.early_stop_counter.update(
                train_loss=train_loss.item(),
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=self.scheduler,
                epoch=epoch,
                end_experiment=(epoch == self.n_epochs)
            )
            if stop_signal == EarlyStopSignal.STOP:
                print("Eraly stopped")
                break

            if epoch % 1 == 0:
                self.val()
                print("Done evaluating")
                self.compute_metrics()
                print("Done computing metrics")
                self.dataset.set_mode('train')
                self.loss.set_mode('train')

        if self.is_main_process:
            print("Training complete.")

    def val(self):
        # switch to validation
        self.dataset.set_mode('val')
        self.loss.set_mode('val')
        self.model.eval()
        if self.is_distributed:
            self.model.module.set_retrieval_module_mode('val')
        else:
            self.model.set_retrieval_module_mode('val')

        if not self.is_distributed:
            dl = DataLoader(self.dataset,
                            batch_size=self.batch_size_val,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_with_optional_cand,
                            drop_last=False)
        else:
            dl = self.get_distributed_dataloader(self.batch_size_val, self.dataset)
            dl.sampler.set_epoch(0)

        self.val_ad_score = []
        # candidates for retrieval (all train windows)
        if self.dataset.use_retrieval:
            X_train = self.dataset.X_train.to(self.device)
            K = self.dataset.num_candidates
            # random draw if requested
            if K > 0:
                indices = np.random.choice(len(X_train), K, replace=False)
                cand_tensor = X_train[indices]  # [K, L, D]
            else:
                cand_tensor = None
        else:
            cand_tensor = None

        for x, _, y in dl:
          if x.shape[0] == self.kwargs.exp_batch_size:
            # for each mask in self.val_masks
            print(x.shape)
            batch_scores = []
            for mask in self.val_masks:
                data = apply_mask(
                    x, p_mask=0.0,
                    eval_mode=True,
                    device=self.device,
                    mask_matrix=~mask
                )
                if cand_tensor is not None:
                    out = self.model(data['masked_tensor'], cand_tensor)
                else:
                    out = self.model(data['masked_tensor'])
                per_sample = self.loss.compute_per_sample(
                    output=out,
                    ground_truth=data['ground_truth'],
                    #num_or_cat=self.dataset.num_or_cat,
                    mask_matrix=data['mask_matrix']
                )
                batch_scores.append(per_sample)
    

            # stack over all masks → [B, n_masks]
            stacked = torch.stack(batch_scores, dim=0).t()
            # normalize & aggregate as before
            if self.kwargs.exp_normalize_ad_loss:
                mn, mx = stacked.min(1)[0], stacked.max(1)[0]
                stacked = (stacked - mn.unsqueeze(1)) / (mx - mn).unsqueeze(1)
            if self.kwargs.exp_aggregation == 'sum':
                final_score = stacked.sum(1)
            else:
                final_score = stacked.max(1)[0]
            # collect (score, label)
            self.val_ad_score.append(torch.stack([final_score,  y.view(-1).float()], dim=1))

        # concatenate and gather across ranks if needed
        self.val_ad_score = torch.cat(self.val_ad_score, dim=0)
        if self.is_distributed:
            gathered = [torch.zeros_like(self.val_ad_score) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, self.val_ad_score)
            self.val_ad_score = torch.cat(gathered, dim=0)

        # split into scores & labels on CPU
        scores = self.val_ad_score[:,0].cpu().detach()
        labels = self.val_ad_score[:,1].cpu().detach().int()
        self.sum_score_ad = (scores, labels)

    def compute_metrics(self):
        scores, labels = self.sum_score_ad
        # normalize
        scores = (scores - scores.mean()) / scores.std()
        ap   = AveragePrecision(task="binary")(scores, labels)
        auc  = AUROC(task="binary")(scores, labels)
        thresh = np.percentile(scores.numpy(), self.dataset.ratio)
        preds  = (scores.numpy() >= thresh).astype(int)
        _, _, f1, _ = prf(labels.numpy(), preds, average='binary')

        result = {
            'seed': self.kwargs.iter_seed,
            'F1': f1,
            'ap': ap,
            'auc': auc
        }
        if self.is_main_process:
            fname = f"results_ratiof1_epoch_{self.epoch}_seed_{self.kwargs.iter_seed}.pkl"
            with open(os.path.join(self.res_dir, fname), 'wb') as f:
                pickle.dump(result, f)
            print("Metrics:", result)
        return result

    def get_distributed_dataloader(self, batchsize, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank
        )
        dl = DataLoader(
            dataset=dataset,
            batch_size=batchsize,
            sampler=sampler,
            num_workers=self.kwargs.data_loader_nprocs,
            pin_memory=False
        )
        if self.is_main_process:
            print("Distributed dataloader ready.")
        return dl
