""" Convenience function for standard architectures with custom non-linearities. """

import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as torchmodels
import models.resnet
import models.vgg


standard_models = torchmodels.__dict__
standard_models.update({f"{name}-custom": model for name, model in models.resnet.__dict__.items() if name.startswith("resnet")})
standard_models.update({f"{name}-custom": model for name, model in models.vgg.__dict__.items() if name.startswith("vgg")})


class LearnableNonLinearity(nn.Module):
    """ Generic non-linearity module with per-channel learnable gain/bias. """

    depth = 0  # Global counter for when the non-linearity depends on the depth.

    def __init__(self, num_channels, non_linearity_name, init_gain=1.0, init_bias=0.0):
        super().__init__()

        if isinstance(non_linearity_name, str):
            self.non_linearity = non_linearity_name
        elif isinstance(non_linearity_name, list):
            self.non_linearity = non_linearity_name[self.depth % len(non_linearity_name)]
            self.__class__.depth += 1
        else:
            assert False

        self.gain = nn.Parameter(torch.full((num_channels,), fill_value=init_gain), requires_grad=self.non_linearity in ["sigmoid", "pow"])
        self.bias = nn.Parameter(torch.full((num_channels,), fill_value=init_bias), requires_grad=self.non_linearity in ["absdead, thresh", "pow"])

    def extra_repr(self) -> str:
        return f"num_channels={self.bias.shape[0]}, non_linearity={self.non_linearity}"

    def forward(self, x):
        if self.non_linearity == "relu":
            return func.relu(x)
        elif self.non_linearity == "abs":
            return torch.abs(x)
        elif self.non_linearity == "absdead":
            bias = self.bias[(slice(None),) + (None,) * (x.ndim - 2)]
            return func.relu(x - bias) + func.relu(-x - bias)
        elif self.non_linearity == "thresh":
            bias = self.bias[(slice(None),) + (None,) * (x.ndim - 2)]
            return func.relu(x - bias) - func.relu(-x - bias)
        elif self.non_linearity == "tanh":
            return func.tanh(x)
        elif self.non_linearity == "sigmoid":
            return func.sigmoid(self.gain[:, None, None] * x.abs() + self.bias[:, None, None]) / (x.abs() + 1e-6) * x
        elif self.non_linearity == "pow":
            t = self.bias[:, None, None].exp() * x.abs() ** self.gain[:, None, None].exp()
            return t / ((1 + t) * (x.abs() + 1e-6)) * x
        elif self.non_linearity == "powfixed":
            return x / (1 + x.abs())
        else:
            assert False