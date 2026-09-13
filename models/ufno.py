"""Three-dimensional U-FNO baseline.

The U-Fourier layer follows Wen et al. (2022): a spectral convolution,
pointwise linear map, and optional mini U-Net are evaluated in parallel.
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv
from torch import nn


class _ConvBlock3D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )


class _UpBlock3D(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )


class MiniUNet3D(nn.Module):
    """Three-level U-Net branch used inside a U-Fourier layer."""

    downsampling_factor = 8

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder1 = _ConvBlock3D(
            channels, channels, stride=2, dropout=dropout
        )
        self.encoder2 = nn.Sequential(
            _ConvBlock3D(channels, channels, stride=2, dropout=dropout),
            _ConvBlock3D(channels, channels, stride=1, dropout=dropout),
        )
        self.encoder3 = nn.Sequential(
            _ConvBlock3D(channels, channels, stride=2, dropout=dropout),
            _ConvBlock3D(channels, channels, stride=1, dropout=dropout),
        )

        self.decoder2 = _UpBlock3D(channels, channels)
        self.decoder1 = _UpBlock3D(2 * channels, channels)
        self.decoder0 = _UpBlock3D(2 * channels, channels)
        self.output = nn.Conv3d(
            2 * channels, channels, kernel_size=3, padding=1
        )

    @classmethod
    def _pad_to_valid_shape(cls, x: torch.Tensor):
        depth, height, width = x.shape[-3:]
        pad_depth = (-depth) % cls.downsampling_factor
        pad_height = (-height) % cls.downsampling_factor
        pad_width = (-width) % cls.downsampling_factor
        padded = F.pad(
            x,
            (0, pad_width, 0, pad_height, 0, pad_depth),
        )
        return padded, (depth, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, original_shape = self._pad_to_valid_shape(x)

        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(encoded1)
        encoded3 = self.encoder3(encoded2)

        decoded2 = self.decoder2(encoded3)
        decoded1 = self.decoder1(torch.cat((encoded2, decoded2), dim=1))
        decoded0 = self.decoder0(torch.cat((encoded1, decoded1), dim=1))
        output = self.output(torch.cat((x, decoded0), dim=1))

        depth, height, width = original_shape
        return output[..., :depth, :height, :width]


class UFourierLayer3D(nn.Module):
    """Parallel spectral, pointwise, and optional U-Net transformations."""

    def __init__(
        self,
        channels: int,
        n_modes: Sequence[int],
        *,
        use_unet: bool,
        unet_dropout: float,
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv(
            in_channels=channels,
            out_channels=channels,
            n_modes=tuple(n_modes),
        )
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        self.unet = (
            MiniUNet3D(channels, dropout=unet_dropout)
            if use_unet
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.spectral(x) + self.pointwise(x)
        if self.unet is not None:
            output = output + self.unet(x)
        return output


class UFNO3D(nn.Module):
    """Channel-first 3D U-FNO with residual next-state prediction."""

    def __init__(
        self,
        n_modes: Sequence[int],
        in_channels: int,
        out_channels: int,
        n_layers: int,
        hidden_channels: int,
        n_unet_layers: int,
        lifting_channels: int = 128,
        projection_channels: int = 128,
        unet_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(n_modes) != 3:
            raise ValueError(f"UFNO3D requires three mode counts, got {n_modes}")
        if not 0 <= n_unet_layers <= n_layers:
            raise ValueError(
                "n_unet_layers must be between zero and n_layers, "
                f"got {n_unet_layers} and {n_layers}"
            )
        if out_channels > in_channels:
            raise ValueError(
                "Residual output requires out_channels <= in_channels, "
                f"got {out_channels} and {in_channels}"
            )

        self.out_channels = out_channels
        self.n_layers = n_layers
        self.n_unet_layers = n_unet_layers

        self.lifting = nn.Sequential(
            nn.Conv3d(in_channels + 3, lifting_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(lifting_channels, hidden_channels, kernel_size=1),
        )

        first_unet_layer = n_layers - n_unet_layers
        self.layers = nn.ModuleList(
            [
                UFourierLayer3D(
                    hidden_channels,
                    n_modes,
                    use_unet=layer_idx >= first_unet_layer,
                    unet_dropout=unet_dropout,
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.activation = nn.GELU()

        self.projection = nn.Sequential(
            nn.Conv3d(hidden_channels, projection_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(projection_channels, out_channels, kernel_size=1),
        )

    @staticmethod
    def _append_coordinate_grid(x: torch.Tensor) -> torch.Tensor:
        batch_size, _, depth, height, width = x.shape
        coordinate_vectors = (
            torch.linspace(0, 1, depth, device=x.device, dtype=x.dtype),
            torch.linspace(0, 1, height, device=x.device, dtype=x.dtype),
            torch.linspace(0, 1, width, device=x.device, dtype=x.dtype),
        )
        coordinate_shapes = (
            (1, 1, depth, 1, 1),
            (1, 1, 1, height, 1),
            (1, 1, 1, 1, width),
        )

        grids = [
            vector.view(shape).expand(batch_size, 1, depth, height, width)
            for vector, shape in zip(coordinate_vectors, coordinate_shapes)
        ]
        return torch.cat((x, *grids), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x[:, : self.out_channels]
        x = self.lifting(self._append_coordinate_grid(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.projection(x) + residual
