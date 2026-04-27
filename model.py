from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
from torch import Tensor, nn


ActivationFactory = Callable[[], nn.Module]


def _get_activation(name: str) -> ActivationFactory:
    normalized = name.lower().strip()
    if normalized == "relu":
        return nn.ReLU
    if normalized == "gelu":
        return nn.GELU
    if normalized == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation: {name}")


@dataclass
class DynamicMLPConfig:
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str = "relu"


class DynamicHiddenLayer(nn.Module):
    """Linear + nonlinearity + neuron mask for targeted pruning/disable."""

    def __init__(self, in_features: int, out_features: int, activation_factory: ActivationFactory) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation_factory()
        self.register_buffer("mask", torch.ones(out_features))

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def forward(self, x: Tensor) -> Tensor:
        h = self.activation(self.linear(x))
        return h * self.mask.unsqueeze(0)

    @torch.no_grad()
    def grow(self, n_new: int, init_std: float = 0.02) -> None:
        if n_new <= 0:
            return

        old_linear = self.linear
        old_out = old_linear.out_features
        new_out = old_out + n_new

        new_linear = nn.Linear(old_linear.in_features, new_out, device=old_linear.weight.device, dtype=old_linear.weight.dtype)
        new_linear.weight[:old_out].copy_(old_linear.weight)
        new_linear.bias[:old_out].copy_(old_linear.bias)
        # Bootstrap new neurons from high-norm existing units, then perturb.
        row_norms = torch.norm(old_linear.weight, p=2, dim=1)
        source_idx = torch.argsort(row_norms, descending=True)
        if source_idx.numel() == 0:
            nn.init.normal_(new_linear.weight[old_out:], mean=0.0, std=init_std)
            nn.init.zeros_(new_linear.bias[old_out:])
        else:
            for new_offset in range(n_new):
                src = int(source_idx[new_offset % source_idx.numel()].item())
                new_linear.weight[old_out + new_offset].copy_(old_linear.weight[src])
                new_linear.weight[old_out + new_offset].add_(torch.randn_like(old_linear.weight[src]) * init_std)
                new_linear.bias[old_out + new_offset].copy_(old_linear.bias[src])

        self.linear = new_linear

        new_mask = torch.ones(new_out, device=self.mask.device, dtype=self.mask.dtype)
        new_mask[:old_out].copy_(self.mask)
        self.mask = new_mask

    @torch.no_grad()
    def disable_neurons(self, neuron_indices: Sequence[int]) -> None:
        if not neuron_indices:
            return
        valid = [idx for idx in neuron_indices if 0 <= idx < self.out_features]
        if not valid:
            return

        idx_tensor = torch.tensor(valid, device=self.linear.weight.device, dtype=torch.long)
        self.mask[idx_tensor] = 0.0
        self.linear.weight[idx_tensor] = 0.0
        self.linear.bias[idx_tensor] = 0.0


class DynamicMLP(nn.Module):
    """Dynamically mutable fully-connected network with targeted hidden-layer growth/pruning."""

    def __init__(self, config: DynamicMLPConfig) -> None:
        super().__init__()
        if len(config.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer")

        self.config = config
        activation_factory = _get_activation(config.activation)

        self.hidden_layers = nn.ModuleList()
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            self.hidden_layers.append(DynamicHiddenLayer(prev_dim, hidden_dim, activation_factory))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, config.output_dim)

    @property
    def hidden_dims(self) -> List[int]:
        return [layer.out_features for layer in self.hidden_layers]

    @property
    def num_trainable_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def architecture_summary(self) -> str:
        hidden = ",".join(str(width) for width in self.hidden_dims)
        return (
            f"hidden_dims=[{hidden}] | "
            f"output_layer(in={self.output_layer.in_features},out={self.output_layer.out_features}) | "
            f"trainable_params={self.num_trainable_parameters}"
        )

    def forward(self, x: Tensor, return_hidden: bool = False) -> Tensor | tuple[Tensor, List[Tensor]]:
        hidden_activations: List[Tensor] = []
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
            hidden_activations.append(h)
        logits = self.output_layer(h)
        if return_hidden:
            return logits, hidden_activations
        return logits

    @property
    def expects_flatten_input(self) -> bool:
        return True

    @torch.no_grad()
    def grow_layer(self, layer_idx: int, n_new: int, init_std: float = 0.02) -> None:
        if not (0 <= layer_idx < len(self.hidden_layers)):
            raise IndexError(f"Invalid hidden layer index: {layer_idx}")
        if n_new <= 0:
            return

        target_layer = self.hidden_layers[layer_idx]
        old_out = target_layer.out_features
        target_layer.grow(n_new, init_std=init_std)

        if layer_idx < len(self.hidden_layers) - 1:
            downstream_linear = self.hidden_layers[layer_idx + 1].linear
            self.hidden_layers[layer_idx + 1].linear = self._expand_linear_input(
                downstream_linear,
                n_new=n_new,
                init_std=init_std,
            )
        else:
            self.output_layer = self._expand_linear_input(self.output_layer, n_new=n_new, init_std=init_std)

        self.config.hidden_dims[layer_idx] = old_out + n_new

    @torch.no_grad()
    def prune_layer_neurons(self, layer_idx: int, neuron_indices: Sequence[int]) -> None:
        if not (0 <= layer_idx < len(self.hidden_layers)):
            raise IndexError(f"Invalid hidden layer index: {layer_idx}")
        if not neuron_indices:
            return

        valid = [idx for idx in neuron_indices if 0 <= idx < self.hidden_layers[layer_idx].out_features]
        if not valid:
            return

        self.hidden_layers[layer_idx].disable_neurons(valid)

        idx_tensor = torch.tensor(valid, device=self.output_layer.weight.device, dtype=torch.long)
        if layer_idx < len(self.hidden_layers) - 1:
            downstream = self.hidden_layers[layer_idx + 1].linear
            downstream.weight[:, idx_tensor] = 0.0
        else:
            self.output_layer.weight[:, idx_tensor] = 0.0

    @staticmethod
    @torch.no_grad()
    def _expand_linear_input(linear: nn.Linear, n_new: int, init_std: float = 0.02) -> nn.Linear:
        if n_new <= 0:
            return linear

        old_in = linear.in_features
        new_linear = nn.Linear(
            old_in + n_new,
            linear.out_features,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        new_linear.weight[:, :old_in].copy_(linear.weight)
        # Initialize new input columns from strong existing connections plus noise.
        col_norms = torch.norm(linear.weight, p=2, dim=0)
        source_idx = torch.argsort(col_norms, descending=True)
        if source_idx.numel() == 0:
            nn.init.normal_(new_linear.weight[:, old_in:], mean=0.0, std=init_std)
        else:
            for col_offset in range(n_new):
                src = int(source_idx[col_offset % source_idx.numel()].item())
                new_linear.weight[:, old_in + col_offset].copy_(linear.weight[:, src])
                new_linear.weight[:, old_in + col_offset].add_(torch.randn_like(linear.weight[:, src]) * init_std)
        new_linear.bias.copy_(linear.bias)
        return new_linear


@dataclass
class DynamicCNNConfig:
    in_channels: int
    channels: List[int]
    num_classes: int
    activation: str = "relu"
    kernel_size: int = 3
    use_batchnorm: bool = True


class DynamicConvLayer(nn.Module):
    """Conv2d + optional batchnorm + nonlinearity + channel mask for targeted pruning/disable."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation_factory: ActivationFactory,
        *,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.activation = activation_factory()
        self.register_buffer("mask", torch.ones(out_channels))

    @property
    def in_features(self) -> int:
        return int(self.conv.in_channels)

    @property
    def out_features(self) -> int:
        return int(self.conv.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        h = self.activation(h)
        return h * self.mask.view(1, -1, 1, 1)

    @torch.no_grad()
    def grow(self, n_new: int, init_std: float = 0.02) -> None:
        if n_new <= 0:
            return

        old_conv = self.conv
        old_out = int(old_conv.out_channels)
        new_out = old_out + int(n_new)
        new_conv = nn.Conv2d(
            int(old_conv.in_channels),
            new_out,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=(old_conv.bias is not None),
            padding_mode=old_conv.padding_mode,
            device=old_conv.weight.device,
            dtype=old_conv.weight.dtype,
        )

        new_conv.weight[:old_out].copy_(old_conv.weight)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias[:old_out].copy_(old_conv.bias)

        # Initialize new filters from strong existing filters plus noise.
        flat = old_conv.weight.detach().view(old_out, -1)
        row_norms = torch.norm(flat, p=2, dim=1)
        source_idx = torch.argsort(row_norms, descending=True)
        if source_idx.numel() == 0:
            nn.init.normal_(new_conv.weight[old_out:], mean=0.0, std=init_std)
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias[old_out:])
        else:
            for new_offset in range(int(n_new)):
                src = int(source_idx[new_offset % source_idx.numel()].item())
                new_conv.weight[old_out + new_offset].copy_(old_conv.weight[src])
                new_conv.weight[old_out + new_offset].add_(torch.randn_like(old_conv.weight[src]) * init_std)
                if new_conv.bias is not None and old_conv.bias is not None:
                    new_conv.bias[old_out + new_offset].copy_(old_conv.bias[src])

        self.conv = new_conv

        if self.bn is not None:
            old_bn = self.bn
            new_bn = nn.BatchNorm2d(
                new_out,
                eps=old_bn.eps,
                momentum=old_bn.momentum,
                affine=old_bn.affine,
                track_running_stats=old_bn.track_running_stats,
                device=old_bn.weight.device if old_bn.weight is not None else None,
                dtype=old_bn.weight.dtype if old_bn.weight is not None else None,
            )
            if old_bn.affine:
                new_bn.weight[:old_out].copy_(old_bn.weight)
                new_bn.bias[:old_out].copy_(old_bn.bias)
            new_bn.running_mean[:old_out].copy_(old_bn.running_mean)
            new_bn.running_var[:old_out].copy_(old_bn.running_var)
            new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
            self.bn = new_bn

        new_mask = torch.ones(new_out, device=self.mask.device, dtype=self.mask.dtype)
        new_mask[:old_out].copy_(self.mask)
        self.mask = new_mask

    @torch.no_grad()
    def disable_channels(self, channel_indices: Sequence[int]) -> None:
        if not channel_indices:
            return
        valid = [idx for idx in channel_indices if 0 <= idx < self.out_features]
        if not valid:
            return

        idx_tensor = torch.tensor(valid, device=self.conv.weight.device, dtype=torch.long)
        self.mask[idx_tensor] = 0.0
        self.conv.weight[idx_tensor] = 0.0
        if self.conv.bias is not None:
            self.conv.bias[idx_tensor] = 0.0
        if self.bn is not None and self.bn.affine:
            self.bn.weight[idx_tensor] = 0.0
            self.bn.bias[idx_tensor] = 0.0


class DynamicCNN(nn.Module):
    """Dynamically mutable CNN with channel-wise growth/pruning in conv layers."""

    def __init__(self, config: DynamicCNNConfig) -> None:
        super().__init__()
        if len(config.channels) == 0:
            raise ValueError("channels must contain at least one conv layer")

        self.config = config
        activation_factory = _get_activation(config.activation)

        self.hidden_layers = nn.ModuleList()
        prev = int(config.in_channels)
        for ch in config.channels:
            self.hidden_layers.append(
                DynamicConvLayer(
                    prev,
                    int(ch),
                    activation_factory,
                    kernel_size=int(config.kernel_size),
                    use_batchnorm=bool(config.use_batchnorm),
                )
            )
            prev = int(ch)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = nn.Linear(prev, int(config.num_classes))

    @property
    def hidden_dims(self) -> List[int]:
        return [int(layer.out_features) for layer in self.hidden_layers]

    @property
    def num_trainable_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    @property
    def expects_flatten_input(self) -> bool:
        return False

    def architecture_summary(self) -> str:
        chans = ",".join(str(ch) for ch in self.hidden_dims)
        return (
            f"conv_channels=[{chans}] | classifier(in={self.output_layer.in_features},out={self.output_layer.out_features}) | "
            f"trainable_params={self.num_trainable_parameters}"
        )

    def forward(self, x: Tensor, return_hidden: bool = False) -> Tensor | tuple[Tensor, List[Tensor]]:
        hidden_activations: List[Tensor] = []
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
            hidden_activations.append(h)

        h = self.pool(h)
        h = torch.flatten(h, 1)
        logits = self.output_layer(h)
        if return_hidden:
            return logits, hidden_activations
        return logits

    @torch.no_grad()
    def grow_layer(self, layer_idx: int, n_new: int, init_std: float = 0.02) -> None:
        if not (0 <= layer_idx < len(self.hidden_layers)):
            raise IndexError(f"Invalid conv layer index: {layer_idx}")
        if n_new <= 0:
            return

        target_layer: DynamicConvLayer = self.hidden_layers[layer_idx]
        old_out = int(target_layer.out_features)
        target_layer.grow(int(n_new), init_std=init_std)

        if layer_idx < len(self.hidden_layers) - 1:
            downstream: DynamicConvLayer = self.hidden_layers[layer_idx + 1]
            downstream.conv = self._expand_conv_in_channels(downstream.conv, n_new=int(n_new), init_std=init_std)
        else:
            self.output_layer = DynamicMLP._expand_linear_input(self.output_layer, n_new=int(n_new), init_std=init_std)

        self.config.channels[layer_idx] = old_out + int(n_new)

    @torch.no_grad()
    def prune_layer_neurons(self, layer_idx: int, neuron_indices: Sequence[int]) -> None:
        if not (0 <= layer_idx < len(self.hidden_layers)):
            raise IndexError(f"Invalid conv layer index: {layer_idx}")
        if not neuron_indices:
            return

        layer: DynamicConvLayer = self.hidden_layers[layer_idx]
        valid = [idx for idx in neuron_indices if 0 <= idx < layer.out_features]
        if not valid:
            return

        layer.disable_channels(valid)

        idx_tensor = torch.tensor(valid, device=self.output_layer.weight.device, dtype=torch.long)
        if layer_idx < len(self.hidden_layers) - 1:
            downstream: DynamicConvLayer = self.hidden_layers[layer_idx + 1]
            downstream.conv.weight[:, idx_tensor, :, :] = 0.0
        else:
            self.output_layer.weight[:, idx_tensor] = 0.0

    @staticmethod
    @torch.no_grad()
    def _expand_conv_in_channels(conv: nn.Conv2d, n_new: int, init_std: float = 0.02) -> nn.Conv2d:
        if n_new <= 0:
            return conv

        old_in = int(conv.in_channels)
        new_conv = nn.Conv2d(
            old_in + int(n_new),
            int(conv.out_channels),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )

        new_conv.weight[:, :old_in, :, :].copy_(conv.weight)

        # Initialize new input-channel slices from strong existing input channels plus noise.
        # Score each input channel by its L2 norm across all output filters.
        channel_norms = torch.norm(conv.weight.detach().view(int(conv.out_channels), old_in, -1), p=2, dim=(0, 2))
        source_idx = torch.argsort(channel_norms, descending=True)
        if source_idx.numel() == 0:
            nn.init.normal_(new_conv.weight[:, old_in:, :, :], mean=0.0, std=init_std)
        else:
            for ch_offset in range(int(n_new)):
                src = int(source_idx[ch_offset % source_idx.numel()].item())
                new_conv.weight[:, old_in + ch_offset, :, :].copy_(conv.weight[:, src, :, :])
                new_conv.weight[:, old_in + ch_offset, :, :].add_(torch.randn_like(conv.weight[:, src, :, :]) * init_std)

        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

        return new_conv
