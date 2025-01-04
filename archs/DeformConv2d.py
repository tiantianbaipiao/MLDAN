import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformConv2d(nn.Module):
    """
    High-Order Offset and Modulation Convolution with nine-directional kernel.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the nine-directional set (fixed at 3x3).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DeformConv2d, self).__init__()
        assert kernel_size == 3, "Only 3x3 kernel size is supported for nine-directional set."

        self.kernel_size = kernel_size
        self.n_directions = kernel_size * kernel_size

        # Standard convolution weights
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        # Offset layers for Δd_i^(1) and Δd_i^(2)
        self.offset_layer_1 = nn.Conv2d(in_channels, 2 * self.n_directions, kernel_size=kernel_size, padding=1)
        self.offset_layer_2 = nn.Conv2d(in_channels, 2 * self.n_directions, kernel_size=kernel_size, padding=1)

        # Modulation scalar layer
        self.modulation_layer = nn.Conv2d(in_channels, self.n_directions, kernel_size=kernel_size, padding=1)

        # Learnable weight α for combining offsets
        self.alpha = nn.Parameter(torch.rand(1))

    def forward(self, x):
        """
        Forward pass of the HighOrderOffsetModulationConv.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        batch_size, _, height, width = x.size()

        # Generate offsets Δd_i^(1) and Δd_i^(2)
        offset_1 = self.offset_layer_1(x)  # Shape: (batch_size, 2 * 9, height, width)
        offset_2 = self.offset_layer_2(x)  # Shape: (batch_size, 2 * 9, height, width)

        # Reshape offsets for grid sampling
        offset_1 = offset_1.view(batch_size, 2, self.n_directions, height, width).permute(0, 3, 4, 2, 1)
        offset_2 = offset_2.view(batch_size, 2, self.n_directions, height, width).permute(0, 3, 4, 2, 1)

        # Generate modulation scalars Δm_i
        modulation = self.modulation_layer(x)  # Shape: (batch_size, 9, height, width)
        modulation = modulation.view(batch_size, self.n_directions, height, width).permute(0, 2, 3, 1)
        modulation = torch.sigmoid(modulation)  # Ensure modulation scalar is in [0, 1]

        # Create a grid for sampling
        grid = self.create_grid(height, width, x.device)  # Shape: (height, width, 9, 2)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # Shape: (batch_size, height, width, 9, 2)

        # Apply high-order offsets to the grid
        sampling_locations = grid + self.alpha * offset_1 + (1 - self.alpha) * offset_2  # Shape: (batch_size, height, width, 9, 2)
        sampling_locations = sampling_locations.view(batch_size, height, width * self.n_directions, 2)

        # Perform grid sampling
        x = x.unsqueeze(1).expand(-1, self.n_directions, -1, -1, -1).reshape(
            batch_size * self.n_directions, -1, height, width
        )
        sampling_locations = sampling_locations.view(batch_size * self.n_directions, height, width, 2)

        sampled_features = F.grid_sample(
            x, sampling_locations, mode='bilinear', padding_mode='zeros', align_corners=False
        )

        # Reshape sampled features back
        sampled_features = sampled_features.view(batch_size, self.n_directions, -1, height, width).permute(0, 2, 3, 4, 1)

        # Apply modulation scalars
        sampled_features = sampled_features * modulation.unsqueeze(1)

        # Sum along the n_directions dimension
        sampled_features = sampled_features.sum(dim=-1)

        # Perform convolution
        out = F.conv2d(sampled_features, self.conv_weight, stride=1, padding=1)

        return out

    @staticmethod
    def create_grid(height, width, device):
        """
        Create a grid of shape (height, width, 9, 2) representing the standard
        3x3 convolutional kernel locations.
        """
        # Define the nine directions (3x3 grid)
        directions = torch.tensor([
            [0, 0], [0, 1], [0, -1],
            [1, 0], [1, 1], [1, -1],
            [-1, 0], [-1, 1], [-1, -1]
        ], dtype=torch.float32, device=device)

        # Create a base grid of shape (height, width, 2)
        base_grid = torch.stack(torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device)
        ), dim=-1)  # Shape: (height, width, 2)

        # Add the nine directions to the base grid
        grid = base_grid.unsqueeze(-2) + directions.view(1, 1, 9, 2)

        return grid
