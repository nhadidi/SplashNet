# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F
import pdb


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int, spec_norm_type: str) -> None:
        super().__init__()
        self.channels = channels
        self.spec_norm_type = spec_norm_type
        if self.spec_norm_type == 'RollingTimeNorm':
            self.spec_norm = RollingTimeNorm(eps = 0.1) 
        elif self.spec_norm_type == 'BatchNorm2d': 
            self.batch_norm = nn.BatchNorm2d(channels)
        elif self.spec_norm_type == 'None':
            self.spec_norm = None
        else:
            raise ValueError(f"Invalid spec_norm_type, got {spec_norm_type}")


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        #assert self.channels == bands * C # probably best to just get rid of this assertion

        if self.spec_norm_type == 'RollingTimeNorm':
            x = self.spec_norm(inputs) 
            return x
        
        elif self.spec_norm_type == 'BatchNorm2d':
            x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
            x = x.reshape(N, bands * C, freq, T)
            x = self.batch_norm(x)
            x = x.reshape(N, bands, C, freq, T)
            return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)
        
        if self.spec_norm is None:
            return inputs



class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
        share_hand_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.share_hand_weights = share_hand_weights

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands if share_hand_weights == False else 1)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        if self.share_hand_weights == False:
            outputs_per_band = [
                mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
            ]
        else:
            outputs_per_band = [
                self.mlps[0](_input) for _input in inputs_per_band
            ]

        return torch.stack(outputs_per_band, dim=self.stack_dim)

    

class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int, share_hand_weights: bool) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.share_hand_weights = share_hand_weights

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width // (2 if share_hand_weights else 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape
        if self.share_hand_weights:
            # Reshape to process both halves in parallel
            inputs = inputs.view(T_in, N*2, C//2)  # [T, 2*N, C//2]

            # T(2N)(C/2) -> (2N)(C/2)T -> (2N)c(w/2)T ### w is half of what it would be without shared weights
            x = inputs.movedim(0, -1).reshape(N*2, self.channels, self.width//2, T_in)
            x = self.conv2d(x)
            x = self.relu(x)
            x = x.reshape(N*2, C//2, -1).movedim(-1, 0)  # (2N)c(w/2)T -> (2N)(C/2)T -> T(2N)(C/2)
                
            # Skip connection afer downsampling 
            T_out = x.shape[0]
            x = x + inputs[-T_out:]
            # Layer norm over C/2
            x = self.layer_norm(x)
            
            return x.view(T_out, N, C)
        
        else:
            # TNC -> NCT -> NcwT
            x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
            x = self.conv2d(x)
            x = self.relu(x)
            x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

            # Skip connection after downsampling
            T_out = x.shape[0]
            x = x + inputs[-T_out:]

            # Layer norm over C
            return self.layer_norm(x)  # TNC
        

class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        share_hand_weights (bool): whether to share weights across hands
            and treat each hand independently
    """
    def __init__(self, num_features: int, share_hand_weights: bool) -> None:
        super().__init__()
        self.num_features = num_features
        self.share_hand_weights = share_hand_weights

        if self.share_hand_weights:
            half_features = num_features // 2
            self.fc_block = nn.Sequential(
                nn.Linear(half_features, half_features),
                nn.ReLU(),
                nn.Linear(half_features, half_features)
            )
            self.layer_norm = nn.LayerNorm(half_features)
        else:
            self.fc_block = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, num_features)
            )
            self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.share_hand_weights:
            # Reshape to process both halves in parallel
            T, N, C = inputs.shape
            x = inputs.view(T, N*2, C//2)  # [T, 2*N, C//2]
            
            # Process both halves simultaneously
            x = self.fc_block(x) + x  # Residual connection
            x = self.layer_norm(x)
            
            # Restore original shape
            return x.view(T, N, C)
        else:
            x = self.fc_block(inputs)
            return self.layer_norm(x + inputs)
        

class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        share_hand_weights: bool = False,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width, share_hand_weights),
                    TDSFullyConnectedBlock(num_features, share_hand_weights),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class RollingTimeNorm(nn.Module):
    r"""Causally normalize a 5D tensor (T, N, bands, channels, freq) along the time axis.

    For each sample (N) and for every (band, channel, frequency) location, this module
    normalizes the data as follows:

    - For the first `warmup` time steps (default 125), the statistics (mean and standard deviation)
      are computed over the entire warmup period (i.e. time indices 0 to 124) and are used for
      every time step in that period.
    - For any subsequent time step \(t \geq \texttt{warmup}\), the statistics are computed over
      all time steps from 0 up through \(t\). For example, at the 300th time step the normalization
      uses the statistics from the first 300 time steps.

    A small epsilon is added inside the square root for numerical stability.
    
    Args:
        warmup (int): Number of time bins to use for warmup (default: 125).
        eps (float): A small constant added for numerical stability.
    """
    def __init__(self, warmup: int = 125, eps: float = 1e-5):
        super().__init__()
        self.warmup = warmup
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (T, N, bands, channels, freq).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        T, N, bands, channels, freq = x.shape
        device, dtype = x.device, x.dtype

        # Compute cumulative sum and cumulative sum of squares along time (dim=0).
        # These have shape (T, N, bands, channels, freq).
        cumsum = x.cumsum(dim=0)
        cumsum2 = (x ** 2).cumsum(dim=0)

        # Create a denominator tensor: [1, 2, 3, ..., T] reshaped for broadcasting.
        time_range = torch.arange(1, T + 1, device=device, dtype=dtype).view(T, 1, 1, 1, 1)

        # Compute the cumulative mean and mean-of-squares.
        cum_mean = cumsum / time_range
        cum_mean2 = cumsum2 / time_range
        cum_std = torch.sqrt(cum_mean2 - cum_mean ** 2 + self.eps)

        # For the warmup period, override the cumulative stats with those computed from
        # the entire warmup period.
        if T >= self.warmup:
            # Use the statistics at time index (warmup - 1) which uses the first 'warmup' time steps.
            warmup_mean = cum_mean[self.warmup - 1]  # shape: (N, bands, channels, freq)
            warmup_mean2 = cum_mean2[self.warmup - 1]
            warmup_std = torch.sqrt(warmup_mean2 - warmup_mean ** 2 + self.eps)

            # For t in [0, warmup), assign the warmup statistics.
            cum_mean[:self.warmup] = warmup_mean.unsqueeze(0).expand(self.warmup, N, bands, channels, freq)
            cum_std[:self.warmup] = warmup_std.unsqueeze(0).expand(self.warmup, N, bands, channels, freq)
        else:
            # If the entire sequence is shorter than the warmup period,
            # use the stats computed from all available time steps.
            warmup_mean = cum_mean[-1]
            warmup_std = torch.sqrt(cum_mean2[-1] - warmup_mean ** 2 + self.eps)
            cum_mean[:] = warmup_mean
            cum_std[:] = warmup_std

        # Normalize: For each time step t, subtract the corresponding mean and divide by std.
        return (x - cum_mean) / cum_std
    