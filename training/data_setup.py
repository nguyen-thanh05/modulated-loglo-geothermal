import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.constants import TEST_INDEX, TRAIN_INDEX
from training.dataset import ARDataset
from training.utils import seed_worker


@dataclass
class DataBundle:
    dataset: ARDataset
    train_loader: DataLoader
    test_loader: DataLoader
    loader_gen: torch.Generator


def build_data_loaders(run_config):
    data_path = run_config.data.path
    k_max = run_config.training.pushforward_k_max

    dataset = ARDataset(
        np.load(os.path.join(data_path, 'all_temp_formation.npy')),
        np.load(os.path.join(data_path, 'all_temp_frac.npy')),
        np.load(os.path.join(data_path, 'all_pres_formation.npy')),
        np.load(os.path.join(data_path, 'all_pres_frac.npy')),
        np.load(os.path.join(data_path, 'all_action.npy')),
        np.load(os.path.join(data_path, 'all_por_matrix.npy')),
        np.load(os.path.join(data_path, 'all_por_frac.npy')),
        np.load(os.path.join(data_path, 'all_perm_matrix.npy')),
        np.load(os.path.join(data_path, 'all_perm_frac.npy')),
        np.load(os.path.join(data_path, 'all_energyrate_bhp.npy')),
        k_max=k_max,
    )

    print("Dataset:", dataset.n_trajectories, "trajectories, heterogeneous")

    train_ds = torch.utils.data.Subset(dataset, TRAIN_INDEX)
    test_ds = torch.utils.data.Subset(dataset, TEST_INDEX)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(run_config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=run_config.training.batch_size,
        shuffle=True,
        generator=loader_gen,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=run_config.training.test_batch_size,
        shuffle=True,
        generator=loader_gen,
        worker_init_fn=seed_worker,
    )

    return DataBundle(
        dataset=dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        loader_gen=loader_gen,
    )
