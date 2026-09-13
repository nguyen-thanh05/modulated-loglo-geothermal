import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64,
                 depth=3, channel_multipliers=None, **kwargs):
        super().__init__()
        h = hidden_channels
        self.out_channels = out_channels
        self.depth = depth

        if channel_multipliers is None:
            channel_multipliers = [2 ** i for i in range(depth + 1)]
        assert len(channel_multipliers) == depth + 1, \
            f"channel_multipliers length {len(channel_multipliers)} != depth+1 ({depth + 1})"

        self.encoders = nn.ModuleList()
        self.encoders.append(ConvBlock(in_channels, h * channel_multipliers[0]))
        for i in range(1, depth):
            self.encoders.append(
                ConvBlock(h * channel_multipliers[i - 1], h * channel_multipliers[i]))

        self.bottleneck = ConvBlock(
            h * channel_multipliers[depth - 1], h * channel_multipliers[depth])

        self.pool = nn.MaxPool3d(2)

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.upsamplers.append(
                nn.ConvTranspose3d(
                    h * channel_multipliers[i + 1],
                    h * channel_multipliers[i],
                    kernel_size=2, stride=2))
            self.decoders.append(
                ConvBlock(
                    h * channel_multipliers[i] * 2,
                    h * channel_multipliers[i]))

        self.final = nn.Conv3d(h * channel_multipliers[0], out_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        x_skip = x[:, :self.out_channels]

        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for i, (up, dec) in enumerate(zip(self.upsamplers, self.decoders)):
            skip_idx = self.depth - 1 - i
            h = dec(torch.cat([up(h), skips[skip_idx]], dim=1))

        return self.final(h) + x_skip
