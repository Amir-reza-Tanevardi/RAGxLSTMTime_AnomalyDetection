

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

# Supported datasets
DSETS = [
    'ettm1', 'Solar', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'creditcard',
    'ettm2', 'etth1', 'etth2', 'electricity', 'traffic', 'illness', 'weather',
    'exchange', 'anomaly_detection', 'nab'
]

def get_dls(params):
    if not hasattr(params, 'use_time_features'):
        params.use_time_features = True

    root_path = '/content/'
    size = [params.context_points, 0, params.target_points]

    if params.dset == 'ettm1':
        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm1.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'anomaly_detection':
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'TimeSeries.csv',
                'label_path': 'labelsTimeSeries.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'nab':
        dls = DataLoaders(
            datasetCls=Dataset_NAB,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ec2_cpu_utilization_24ae8d.csv',
                'label_path': 'combined_labels.json',  # remove if labels are not used
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'Solar':
        dls = DataLoaders(
            datasetCls=Dataset_Solar,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'solar_AL.txt',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
        dls = DataLoaders(
            datasetCls=Dataset_PEMS,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': f'{params.dset}.npz',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset == 'ettm2':
        dls = DataLoaders(
            datasetCls=Dataset_ETT_minute,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ETTm2.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset in ['etth1', 'etth2']:
        dls = DataLoaders(
            datasetCls=Dataset_ETT_hour,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': f'{params.dset.upper()}.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    elif params.dset in ['electricity', 'traffic', 'weather', 'illness', 'exchange']:
        filename_map = {
            'electricity': 'electricity.csv',
            'traffic': 'traffic.csv',
            'weather': 'weather.csv',
            'illness': 'national_illness.csv',
            'exchange': 'exchange_rate.csv'
        }
        dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
                'root_path': root_path,
                'data_path': filename_map[params.dset],
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
        )

    else:
        raise ValueError(f"Unrecognized dataset `{params.dset}`. Supported datasets: {DSETS}")

    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls