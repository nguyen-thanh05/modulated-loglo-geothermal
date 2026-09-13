import numpy as np
import torch
from torch.utils.data import Dataset

from training.constants import BHP_MAX, BHP_MIN, ENERGY_MAX


class ARDataset(Dataset):
    def __init__(
        self,
        temp_formation,
        temp_frac,
        pres_formation,
        pres_frac,
        action,
        por_matrix,
        por_frac,
        perm_matrix,
        perm_frac,
        aux_energy_bhp=None,
        k_max=5,
    ):
        self.temp_formation = temp_formation
        self.temp_frac = temp_frac
        self.pres_formation = pres_formation
        self.pres_frac = pres_frac
        self.action = action
        self.por_matrix = por_matrix
        self.por_frac = por_frac
        self.perm_matrix = perm_matrix
        self.perm_frac = perm_frac
        self.aux_energy_bhp = aux_energy_bhp

        self.temp_min = 20.0
        self.temp_max = 185.0
        self.action_min = 0.0
        self.action_max = 5000.0
        self.pres_min = 1300.0
        self.pres_max = 70000.0
        self.por_min_matrix = 0.03
        self.por_max_matrix = 0.07
        self.por_min_frac = 0.002
        self.por_max_frac = 0.008
        self.perm_min_matrix = 0.05
        self.perm_max_matrix = 0.12
        self.perm_min_frac = 3.0
        self.perm_max_frac = 190.0
        self.bhp_min = BHP_MIN
        self.bhp_max = BHP_MAX
        self.energy_max = ENERGY_MAX

        self._temp_range = self.temp_max - self.temp_min
        self._pres_range = self.pres_max - self.pres_min
        self._action_range = self.action_max - self.action_min
        self._bhp_range = self.bhp_max - self.bhp_min
        self._energy_log_denom = float(np.log1p(self.energy_max))

        self.k_max = k_max
        self.n_trajectories = len(temp_formation)
        self.n_timesteps = 156

    def __len__(self):
        return self.n_trajectories

    def _to_tensor(self, np_array):
        return torch.from_numpy(np_array.copy())

    def _state_at(self, idx, step):
        tf = (self._to_tensor(self.temp_formation[idx, step]) - self.temp_min) / self._temp_range
        tfr = (self._to_tensor(self.temp_frac[idx, step]) - self.temp_min) / self._temp_range
        pf = (self._to_tensor(self.pres_formation[idx, step]) - self.pres_min) / self._pres_range
        pfr = (self._to_tensor(self.pres_frac[idx, step]) - self.pres_min) / self._pres_range
        return torch.stack([tf, tfr, pf, pfr], dim=0).float()

    def _action_at(self, idx, step):
        a = (self._to_tensor(self.action[idx, step]) - self.action_min) / self._action_range
        return a.float()

    def _static_at(self, idx):
        pm = (self._to_tensor(self.por_matrix[idx]) - self.por_min_matrix) / (self.por_max_matrix - self.por_min_matrix)
        pf = (self._to_tensor(self.por_frac[idx]) - self.por_min_frac) / (self.por_max_frac - self.por_min_frac)
        km = (self._to_tensor(self.perm_matrix[idx]) - self.perm_min_matrix) / (self.perm_max_matrix - self.perm_min_matrix)
        kf = (self._to_tensor(self.perm_frac[idx]) - self.perm_min_frac) / (self.perm_max_frac - self.perm_min_frac)
        return torch.stack([pm, pf, km, kf], dim=0).float()

    def _aux_at(self, idx, step):
        if self.aux_energy_bhp is None:
            return torch.zeros(16, dtype=torch.float32)
        aux = self._to_tensor(self.aux_energy_bhp[idx, step]).float()
        bhp = (aux[:9] - self.bhp_min) / self._bhp_range
        energy = torch.log1p(aux[9:]) / self._energy_log_denom
        return torch.cat([bhp, energy], dim=0)

    def denormalize_aux(self, aux_normalized):
        """Undo BHP min-max and energy log1p. aux_normalized: (..., 16)."""
        bhp = aux_normalized[..., :9] * self._bhp_range + self.bhp_min
        energy = torch.expm1(aux_normalized[..., 9:] * self._energy_log_denom)
        return bhp, energy

    def __getitem__(self, idx):
        t = np.random.randint(0, self.n_timesteps)

        y_t = self._state_at(idx, t)
        y_tp1 = self._state_at(idx, t + 1)
        action_t = self._action_at(idx, t)

        y_history = []
        action_history = []
        valid_k = []
        for i in range(self.k_max):
            step = t - (self.k_max - i)
            if step >= 0:
                y_history.append(self._state_at(idx, step))
                action_history.append(self._action_at(idx, step))
                valid_k.append(1.0)
            else:
                y_history.append(torch.zeros_like(y_t))
                action_history.append(torch.zeros_like(action_t))
                valid_k.append(0.0)

        y_history = torch.stack(y_history, dim=0)
        action_history = torch.stack(action_history, dim=0)
        valid_k = torch.tensor(valid_k, dtype=torch.float)

        return (
            y_history, action_history, valid_k, y_t, y_tp1, action_t,
            self._static_at(idx),
            self._aux_at(idx, t),
            self._aux_at(idx, t + 1),
        )
