import torch
import torch.nn as nn

from training.constants import (
    N_PRODUCER_WELLS,
    RATE_SLICE_Z,
    WELL_COORDS,
)


class WellAuxHead(nn.Module):
    """Shared per-well MLP readout for BHP and producer energy rate.

    Each well is encoded independently from its D=16 columns of
    (s_t, s_{t+1}, statics), a scalar rate at one z-slice, and a
    producer/injector one-hot. Outputs are packed as
    [BHP x 9 | energy x 7].
    """

    def __init__(
        self,
        state_channels=4,
        static_channels=4,
        depth=16,
        hidden=(128, 64),
        rate_z=RATE_SLICE_Z,
    ):
        super().__init__()
        self.state_channels = state_channels
        self.static_channels = static_channels
        self.depth = depth
        self.rate_z = rate_z

        well_indices = torch.tensor(WELL_COORDS, dtype=torch.long)
        self.register_buffer('well_indices', well_indices)
        self.num_wells = len(WELL_COORDS)
        self.n_producers = N_PRODUCER_WELLS

        well_type = torch.zeros(self.num_wells, 2)
        well_type[:self.n_producers, 0] = 1.0
        well_type[self.n_producers:, 1] = 1.0
        self.register_buffer('well_type', well_type)

        per_well_dim = (
            depth * (2 * state_channels + static_channels) + 1 + 2
        )
        layers = []
        in_dim = per_well_dim
        for hidden_dim in hidden:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def _extract_columns(self, tensor):
        """(B, C, D, H, W) -> (B, n_wells, C * D)."""
        wx = self.well_indices[:, 0]
        wy = self.well_indices[:, 1]
        cols = tensor[:, :, :, wx, wy]
        batch, channels, depth, n_wells = cols.shape
        return cols.permute(0, 3, 1, 2).reshape(batch, n_wells, channels * depth)

    def forward(self, y_t, y_tp1, static, rate):
        """Predict packed aux from well columns.

        Args:
            y_t: (B, 4, D, H, W) state at t (GT, noisy, or predicted).
            y_tp1: (B, 4, D, H, W) predicted state at t+1.
            static: (B, 4, D, H, W) porosity and permeability.
            rate: (B, D, H, W) signed well-rate channel.
        Returns:
            (B, 16) packed [BHP x 9 | energy x 7].
        """
        wx = self.well_indices[:, 0]
        wy = self.well_indices[:, 1]
        batch = y_t.shape[0]

        y_t_cols = self._extract_columns(y_t)
        y_tp1_cols = self._extract_columns(y_tp1)
        static_cols = self._extract_columns(static)
        well_rate = rate[:, self.rate_z, wx, wy].unsqueeze(-1)
        well_type = self.well_type.unsqueeze(0).expand(batch, -1, -1)

        per_well = torch.cat(
            [y_t_cols, y_tp1_cols, static_cols, well_rate, well_type],
            dim=-1,
        )
        well_out = self.mlp(per_well)
        bhp = well_out[:, :, 0]
        energy = well_out[:, :self.n_producers, 1]
        return torch.cat([bhp, energy], dim=-1)
