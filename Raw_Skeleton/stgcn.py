"""
Spatio-Temporal Graph Convolutional Network (STGCN) for skeleton-based gait analysis.
Input: (B, N_gaits, T, 96) with 32 landmarks × 3 (x,y,z). Output: (B, 1) POMA regression.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_default_skeleton_adjacency(num_nodes=32):
    """
    Build a default normalized adjacency for 32 kinematic markers.
    Topology: spine chain (4-5-6-7), head (0-1-2-3), plus connections from spine to limbs.
    Uses symmetric normalization: D^{-1/2} (A + I) D^{-1/2}.
    """
    A = torch.eye(num_nodes)
    # Spine: C7(4)-T10(5)-CLAV(6)-STRN(7)
    edges = [(4, 5), (5, 6), (6, 7), (4, 6), (5, 7)]
    # Head to spine
    edges += [(0, 4), (1, 4), (2, 4), (3, 4)]
    # Chain adjacent markers (generic connectivity)
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
    for (i, j) in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Symmetric normalization
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
    A_norm = D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
    return A_norm


class SpatialGraphConv(nn.Module):
    """Graph convolution over skeleton: Y = sigma(A @ X @ W)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, A):
        # x: (B, T, N, C_in), A: (N, N)
        # (B,T,N,C) -> (B,T,N,C): aggregate neighbors via A
        x = torch.einsum("ij,btjc->btic", A, x)
        return self.linear(x)


class TemporalConvBlock(nn.Module):
    """1D convolution along time with residual."""

    def __init__(self, channels, kernel_size=9):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(channels, channels, (kernel_size, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # x: (B, T, N, C) -> (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        out = self.bn(self.conv(x))
        out = out.permute(0, 2, 3, 1)
        return out


class STBlock(nn.Module):
    """Spatial graph conv + temporal conv block with residual."""

    def __init__(self, in_channels, out_channels, kernel_size=9):
        super().__init__()
        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal = TemporalConvBlock(out_channels, kernel_size)
        self.temporal_bn = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x, A):
        # x: (B, T, N, C)
        res = x
        out = self.spatial(x, A)
        # (B,T,N,C) -> (B,C,T,N) for BN
        out = out.permute(0, 3, 1, 2)
        out = F.relu(self.spatial_bn(out))
        out = out.permute(0, 2, 3, 1)
        out = self.temporal(out)
        out = out.permute(0, 3, 1, 2)
        out = self.temporal_bn(out)
        out = out.permute(0, 2, 3, 1)
        if self.downsample is not None:
            res = self.downsample(res)
        return F.relu(out + res)


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for POMA (stroke severity) regression.
    Input shape: (B, N_gaits, T, 96) — batch, gait windows, time per gait (100), 32*3 channels.
    Output shape: (B, 1).
    """

    def __init__(
        self,
        N_gaits=2,
        num_nodes=32,
        in_channels=3,
        input_dim=96,
        hidden_channels=(64, 64, 128),
        kernel_size=9,
        dropout=0.3,
        output_dim=1,
        adj=None,
    ):
        super().__init__()
        assert input_dim == num_nodes * in_channels, "input_dim must be num_nodes * in_channels (32*3=96)"
        self.N_gaits = N_gaits
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.seq_len = 100 * N_gaits
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout

        if adj is not None:
            self.register_buffer("A", adj.float())
        else:
            self.register_buffer("A", build_default_skeleton_adjacency(num_nodes))

        self.input_embed = nn.Linear(in_channels, hidden_channels[0])

        layers = []
        for i in range(len(hidden_channels) - 1):
            layers.append(
                STBlock(hidden_channels[i], hidden_channels[i + 1], kernel_size=kernel_size)
            )
        self.st_blocks = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels[-1], output_dim)

    def forward(self, x):
        # x: (B, G, T, 96)
        B, G, T, C = x.shape
        # Reshape to (B, G*T, 32, 3)
        x = x.view(B, G * T, self.num_nodes, self.in_channels)
        # (B, T_total, N, C_in)
        x = self.input_embed(x)
        # x: (B, T_total, N, hidden[0])
        for block in self.st_blocks:
            x = block(x, self.A)
        # Global pool over time and nodes
        x = x.mean(dim=(1, 2))
        x = F.gelu(x)
        x = self.dropout(x)
        return self.fc(x)
