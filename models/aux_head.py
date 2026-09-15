import torch
import torch.nn as nn

from training.constants import (
    N_PRODUCER_WELLS,
    RATE_SLICE_Z,
    WELL_COORDS,
)


class WellAuxHead(nn.Module):
    """Shared per-well readout for BHP and producer energy rate.

    Each well is encoded from a 3x3xD patch of (s_t, s_{t+1}, statics),
    mixed by a valid Conv3d, then an MLP. A scalar rate at one z-slice
    and a producer/injector one-hot are concatenated after the conv.
    Outputs are packed as [BHP x 9 | energy x 7].
    """

    PATCH_RADIUS = 1
    CONV_KERNEL = 3

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
        self.field_channels = 2 * state_channels + static_channels
        self.mix_depth = depth - (self.CONV_KERNEL - 1)

        well_indices = torch.tensor(WELL_COORDS, dtype=torch.long)
        self.register_buffer('well_indices', well_indices)
        self.num_wells = len(WELL_COORDS)
        self.n_producers = N_PRODUCER_WELLS

        well_type = torch.zeros(self.num_wells, 2)
        well_type[:self.n_producers, 0] = 1.0
        well_type[self.n_producers:, 1] = 1.0
        self.register_buffer('well_type', well_type)

        self.mix = nn.Sequential(
            nn.Conv3d(
                self.field_channels,
                self.field_channels,
                kernel_size=self.CONV_KERNEL,
                padding=0,
            ),
            nn.GELU(),
        )

        per_well_dim = self.field_channels * self.mix_depth + 1 + 2
        layers = []
        in_dim = per_well_dim
        for hidden_dim in hidden:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2))
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _extract_patches(self, tensor):
        """(B, C, D, H, W) -> (B, n_wells, C, D, 3, 3)."""
        height, width = tensor.shape[-2:]
        wx = self.well_indices[:, 0]
        wy = self.well_indices[:, 1]
        radius = self.PATCH_RADIUS
        if (
            torch.any(wx < radius)
            or torch.any(wx > height - radius - 1)
            or torch.any(wy < radius)
            or torch.any(wy > width - radius - 1)
        ):
            raise AssertionError(
                f"3x3 well patches out of bounds for grid {(height, width)} "
                f"with wells {self.well_indices.tolist()}"
            )

        offsets = torch.arange(
            -radius, radius + 1, device=tensor.device, dtype=wx.dtype,
        )
        hs = wx[:, None, None] + offsets[None, :, None]
        ws = wy[:, None, None] + offsets[None, None, :]
        patches = tensor[:, :, :, hs, ws]
        return patches.permute(0, 3, 1, 2, 4, 5)

    def forward(self, y_t, y_tp1, static, rate):
        """Predict packed aux from 3x3xD well patches.

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

        stacked = torch.cat(
            [
                self._extract_patches(y_t),
                self._extract_patches(y_tp1),
                self._extract_patches(static),
            ],
            dim=2,
        )
        mixed = self.mix(
            stacked.reshape(
                batch * self.num_wells,
                self.field_channels,
                self.depth,
                2 * self.PATCH_RADIUS + 1,
                2 * self.PATCH_RADIUS + 1,
            )
        )
        mixed = mixed.reshape(batch, self.num_wells, -1)

        well_rate = rate[:, self.rate_z, wx, wy].unsqueeze(-1)
        well_type = self.well_type.unsqueeze(0).expand(batch, -1, -1)
        per_well = torch.cat([mixed, well_rate, well_type], dim=-1)
        well_out = self.mlp(per_well)
        bhp = well_out[:, :, 0]
        energy = well_out[:, :self.n_producers, 1]
        return torch.cat([bhp, energy], dim=-1)
