"""Load model, data and corresponding configs. Trigger training (now for time-series data)."""
import os, ast, json
import random
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformer import Model
from retrieval import Retrieval
from loss import Loss
from torch_dataset import TorchDataset
from train import Trainer

from configs import build_parser
from utils.encode_utils import get_torch_dtype
from utils.train_utils import init_optimizer, count_parameters
from utils.log_utils import print_args

from datasets.dict_to_data import DATASET_NAME_TO_DATASET_MAP

# Environment variables set by torch.distributed.launch
if torch.cuda.is_available():
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    WORLD_RANK = int(os.environ['RANK'])

def main(args):

    # 1) initialize distributed if requested
    if args.mp_distributed:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=WORLD_SIZE,
            rank=WORLD_RANK
        )
    
    kwargs = vars(args)
    # 2) Pass sequence length into the dataset constructor
    kwargs.update({'seq_len': args.seq_len})
    dataset = DATASET_NAME_TO_DATASET_MAP[args.data_set](**kwargs)
    
    args.is_batchlearning = (args.exp_batch_size != -1)
    args.iteration = 0
    if (not args.mp_distributed) or (args.mp_distributed and LOCAL_RANK == 0):
        print_args(args)

    metrics = []
    val_duration = []

    for iteration in range(args.exp_n_runs):
        args.iter_seed = args.np_seed + iteration

        # seed everything
        torch.manual_seed(args.torch_seed + iteration)
        np.random.seed(args.iter_seed)
        random.seed(args.iter_seed)

        # ----- distributed / single-GPU branching -----
        if args.mp_distributed:
            device = LOCAL_RANK
            torch.cuda.set_device(device)
            dist_args = {'world_size': WORLD_SIZE, 'rank': WORLD_RANK, 'gpu': device}
        else:
            device = 'cpu' if (args.mp_gpus == 0 or not torch.cuda.is_available()) else 'cuda:0'
            if device != 'cpu':
                torch.cuda.set_device(device)
            # Disable AMP if CPU
            if device == 'cpu':
                args.model_amp = False
                args.full_dataset_cuda = False
            dist_args = None

        args.device = device
        args.iteration = iteration

        # ----- load and preprocess time-series data -----
        dataset.load()
        dataset.load_time_series(seq_len = args.seq_len)  
        # This should:
        #  - read the raw time-indexed data
        #  - normalize / featurize timestamps if needed
        #  - generate rolling windows of length seq_len
        #   (e.g. self.windows = sliding_window(self.data, seq_len))
        #
        # You could factor that into a new `load_time_series()` method.

        # chronological split: train on earliest windows, validate on the tail
        dataset.split_time_series_train_val(
            train_ratio=args.train_ratio,
            seed=args.np_seed,
            iteration=iteration
        )
        # This replaces split_train_val; implements a non‑random, time‑based split.

        # wrap into TorchDataset (will index windows)
        torchdataset = TorchDataset(
            dataset=dataset,
            mode='train',
            seq_len=args.seq_len,
            kwargs=args,
            device=device
        )

        # ----- optional retrieval module (e.g. temporal k‑NN helpers) -----
        if args.exp_retrieval_type.lower() != 'none':
            # unmodified, but seq_len may influence representation dimensions
            if args.exp_retrieval_type.lower() == 'knn':
                retrieval_dict = {
                    'knn_type': args.exp_retrieval_knn_type,
                    'k': args.exp_retrieval_num_helpers,
                    'metric': args.exp_retrieval_metric
                }
            else:
                assert args.exp_retrieval_agg_location != 'pre-embedding', \
                    "Retrieval location cannot be before embedding for non-KNN types"
                in_dim_ret = args.model_dim_hidden * dataset.D * args.seq_len
                out_dim_ret = args.model_dim_hidden * dataset.D * args.seq_len
                retrieval_dict = {
                    'retrieval_type': args.exp_retrieval_type,
                    'in_dim': in_dim_ret,
                    'out_dim': out_dim_ret,
                    'k': args.exp_retrieval_num_helpers,
                    'normalization_sim': args.exp_retrieval_normalization_sim,
                    'n_features': dataset.D
                }

            retrieval_module = Retrieval(
                type_retrieval=args.exp_retrieval_type,
                num_helpers=args.exp_retrieval_num_helpers,
                retrieval_kwargs=retrieval_dict,
                deterministic_selection=[],
                device=device
            )
        else:
            retrieval_module = None

        # ----- model initialization -----
        model = Model(
            D                = dataset.D,          # total features per time‑step
            seq_len          = args.seq_len,
            hidden_dim       = args.model_dim_hidden,
            num_layers_e       = args.model_num_layers_e,   # we use a single stack
            num_heads_e        = args.model_num_heads_e,
            num_layers_d       = args.model_num_layers_d,   # we use a single stack
            num_heads_d        = args.model_num_heads_d,
            p_dropout        = args.model_hidden_dropout_prob,
            retrieval        = retrieval_module,
            gradient_clipping= args.exp_gradient_clipping,
            layer_norm_eps=13-5,
            args=args,
            device           = device
        )

        torch_dtype = get_torch_dtype(args.model_dtype)
        model = model.to(device).type(torch_dtype)
        print(f'Model has {count_parameters(model)} parameters, '
              f'batch size {args.exp_batch_size}.')

        # optimizer (include retrieval params if non‑knn)
        if args.exp_retrieval_type.lower() in ['none', 'knn']:
            optimizer = init_optimizer(args=args,
                                       model_parameters=model.parameters())
        else:
            optimizer = init_optimizer(
                args=args,
                model_parameters=list(model.parameters())
                                 + list(model.retrieval_module.parameters())
            )
        print(f'Initialized "{args.exp_optimizer}" optimizer.')
        scaler = GradScaler(enabled=args.model_amp)
        if args.model_amp:
            print('Initialized gradient scaler for AMP.')

        # wrap for distributed training
        if args.mp_distributed:
            model = DDP(model, device_ids=[device], find_unused_parameters=True)
            dist.barrier()

        # ----- loss, trainer, and run -----
        loss = Loss(args,
                    is_batchlearning=args.is_batchlearning,
                    device=device)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss=loss,
            scaler=scaler,
            kwargs=args,
            torch_dataset=torchdataset,
            distributed_args=dist_args,
            device=device
        )

        trainer.train()

        # validation
        val_start_time = datetime.now()
        trainer.val()
        val_end_time = datetime.now()
        val_duration.append((val_end_time - val_start_time).total_seconds())

        ratio_metrics = trainer.compute_metrics()
        metrics.append(ratio_metrics)

        if (not args.mp_distributed) or (device == LOCAL_RANK == 0):
            print(ratio_metrics)
            print('Val duration:', val_duration[-1])

    # ----- summarise multi‑run results -----
    if (not args.mp_distributed) or (LOCAL_RANK == 0):
        if len(metrics) > 1:
            f1    = [m['F1'] for m in metrics]
            auc   = [m['auc'] for m in metrics]
            ap    = [m['ap'] for m in metrics]
            ttime = [m['training_time'] for m in metrics]
            vrate = val_duration

            summary = (
                f"Mean F1 ± std over {args.exp_n_runs} runs: {np.mean(f1):.4f} ± {np.std(f1):.4f}\n"
                f"Mean AUC ± std over {args.exp_n_runs} runs: {np.mean(auc):.4f} ± {np.std(auc):.4f}\n"
                f"Mean AP ± std over {args.exp_n_runs} runs: {np.mean(ap):.4f} ± {np.std(ap):.4f}\n"
                f"Avg training time: {np.mean(ttime):.2f}s ± {np.std(ttime):.2f}s\n"
                f"Avg val time: {np.mean(vrate):.2f}s ± {np.std(vrate):.2f}s\n"
            )
            print(summary)

            out_score = {
                'F1': (np.mean(f1), np.std(f1)),
                'AUC': (np.mean(auc), np.std(auc)),
                'AP': (np.mean(ap), np.std(ap)),
                'train_time': (np.mean(ttime), np.std(ttime)),
                'val_time': (np.mean(vrate), np.std(vrate))
            }
            out_path = os.path.join(trainer.res_dir, 'mean_results.json')
            with open(out_path, 'w') as fp:
                json.dump(out_score, fp)


if __name__ == '__main__':
    parser = build_parser()
    # 3) add a --seq_len and --train_ratio to your CLI
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Length of each input time-series window')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Fraction of windows used for training (chronological split)')
    parser.add_argument(
    "--anomaly_map_path",
    type=str,
    default=None,
    help="Path to NAB JSON file that maps each CSV to its anomaly‑timestamp list."
    )

    args = parser.parse_args()
    # args.model_augmentation_bert_mask_prob = ast.literal_eval(
    #     args.model_augmentation_bert_mask_prob
    # )

    main(args)
