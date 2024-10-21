import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class ComplexConv2d(nn.Module):
    """ Conv2D class which also works for complex input, and can initialize its weight to a unitary operator. """
    def __init__(self, input_type: TensorType, complex_weights, out_channels, kernel_size, parseval, quadrature,
                 eigenvectors="", eigenvectors_start=None, eigenvectors_end=None, eigenvalues="",
                 initialization="",
                 ):
        """
        :param input_type:
        :param complex_weights: whether the weights will be complex or real (None defaults to type of input)
        :param out_channels: int or id/"id", or identity convolution
        :param kernel_size:
        :param parseval: whether this module will be subject to Parseval regularization
        :param quadrature: whether this module will be subject to quadrature regularization
        :param eigenvectors: path to eigenvectors .pt file to constrain learning on a subspace (num_eig, dim, 1, 1)
        :param eigenvectors_start, eigenvectors_end: start (included) and end (excluded) of range for subspace
        :param eigenvalues: path to eigenvalues .pt file to initialize weights to Gaussian distribution
        :param initialization: path to .pt file to initialize weight to custom matrix
        """
        super().__init__()

        self.input_type = input_type
        self.in_channels = self.input_type.num_channels
        self.complex_weights = complex_weights if complex_weights is not None else self.input_type.complex
        self.is_identity = out_channels in [id, "id"]

        if eigenvectors:
            eigenvectors = torch.load(eigenvectors)[eigenvectors_start:eigenvectors_end]  # (Cout, Cin, 1, 1)
            self.register_buffer("pre_proj", eigenvectors)
            num_input_channels = eigenvectors_end - eigenvectors_start
        else:
            num_input_channels = self.in_channels

        self.out_channels = num_input_channels if self.is_identity else out_channels
        self.kernel_size = kernel_size
        assert self.kernel_size == 1
        self.output_type = TensorType(self.out_channels, self.input_type.spatial_shape,
                                      complex=self.input_type.complex or self.complex_weights)

        self.parseval = parseval
        self.quadrature = quadrature

        if self.is_identity:
            # Still useful to have a parameter for batched implementation of TriangularComplexConv2d.
            param = torch.eye(num_input_channels)[..., None, None]
            self.register_buffer("param", param)
        elif self.out_channels == 0:  # PyTorch doesn't cleanly handle 0-sized tensors...
            self.register_buffer("param", torch.empty((0, num_input_channels, 1, 1)))
        else:
            shape = (out_channels, num_input_channels, kernel_size, kernel_size)
            if initialization:
                param = torch.load(initialization)
                assert param.shape == shape
            elif eigenvalues:
                # Gaussian initialization:
                assert not self.complex_weights
                eigenvalues = torch.load(eigenvalues)[eigenvectors_start:eigenvectors_end].clip(min=0)  # (num_input_channels,)
                param = torch.normal(mean=torch.zeros(shape),
                                     std=torch.ones(shape) * torch.sqrt(eigenvalues[None, :, None, None]))
            elif self.complex_weights:
                param = unitary_init(shape)
                param = torch.view_as_real(param)
            else:
                param = nn.Conv2d(in_channels=num_input_channels, out_channels=out_channels,
                                  kernel_size=kernel_size).weight.data
                if self.parseval:
                    nn.init.orthogonal_(param)

            self.param = nn.Parameter(param)

    def extra_repr(self) -> str:
        if self.is_identity:
            extra = "is_identity=True"
        else:
            extra = f"out_channels={type_to_str(self.output_type)}, " \
               f"kernel_size={self.kernel_size}, complex_weights={self.complex_weights}, parseval={self.parseval}, " \
               f"quadrature={self.quadrature}"
        return f"in_channels={type_to_str(self.input_type)}, {extra}"

    def forward(self, x):
        if hasattr(self, "pre_proj"):
            assert not self.output_type.complex  # Not sure how this would work
            x = conv2d(x, self.pre_proj, self.output_type.complex)
        if self.is_identity:
            return x
        elif self.out_channels > 0:
            return conv2d(x, self.param, self.output_type.complex)
        else:
            return x.new_empty((x.shape[0], 0) + x.shape[2:])


class TriangularComplexConv2d(nn.Module):
    """ Efficient (batched) implementation of a convolution with block-triangular weights across channels.
    Equivalent to Sequential(TriangularModule(), DiagonalModule(ComplexConv2d)) but faster. """
    def __init__(self, input_type: SplitTensorType, complex_weights, out_channels, kernel_size, parseval, quadrature):
        """
        :param input_type: split tensor type
        :param complex_weights: whether the weights will be complex or real (None defaults to type of input)
        :param out_channels: dictionary, group key -> number of output channels
        :param kernel_size: global kernel size
        :param parseval: whether to apply Parseval regularization on the full triangular convolution matrix
        :param quadrature: whether to apply quadrature regularization on the full triangular convolution matrix
        """
        super().__init__()

        self.input_type = input_type
        self.kernel_size = kernel_size
        self.complex_weights = complex_weights if complex_weights is not None else self.input_type.complex
        self.parseval = parseval
        self.quadrature = quadrature

        self.keys = self.input_type.keys
        in_channels = 0
        submodules = {}
        for key in self.keys:
            in_channels += input_type.groups[key]

            # Parseval/Quadrature is handled by this module, hence we transmit False to submodules.
            submodules[key] = ComplexConv2d(
                input_type=TensorType(in_channels, input_type.spatial_shape, input_type.complex),
                complex_weights=complex_weights, out_channels=out_channels[key],
                kernel_size=self.kernel_size, parseval=False, quadrature=False,
            )
            out_channels[key] = submodules[key].out_channels  # Replaces identity output channels by their number.
        self.submodules = ModuleDict(submodules)
        self.total_in_channels = in_channels
        self.total_out_channels = sum(out_channels.values())

        self.output_type = SplitTensorType(
            groups=out_channels, spatial_shape=next(self.submodules.values().__iter__()).output_type.spatial_shape,
            complex=self.input_type.complex or self.complex_weights,
        )

    def extra_repr(self):
        def format_complex(complex):
            return 'C' if complex else 'R'
        return f"in_channels={self.total_in_channels}{format_complex(self.input_type.complex)}, " \
               f"out_channels={self.total_out_channels}{format_complex(self.output_type.complex)}, " \
               f"kernel_size={self.kernel_size}, complex_weights={self.complex_weights}, parseval={self.parseval}, " \
               f"quadrature={self.quadrature}"

    def full_weights(self):
        """ Returns the full weights, of shape (out_channels, in_channels, kernel_size, kernel_size).
        If complex weights, returns a real view with an additional last dimension of 2. """
        shape = (1, 1) + ((2,) if self.complex_weights else ())
        if len(self.submodules) == 1:  # Slight optimization (also allows in-place update).
            w = self.submodules[0].param
        else:
            w = torch.cat([torch.cat([sub.param, sub.param.new_zeros(
                (sub.param.shape[0], self.total_in_channels - sub.param.shape[1]) + shape)], dim=1)
                           for sub in self.submodules.values() if sub.param.numel() > 0], dim=0)  # deal with 0 input or output case
        return w

    def forward(self, x: SplitTensor) -> SplitTensor:
        x = x.full_view()
        w = self.full_weights()
        y = conv2d(x, w, self.output_type.complex)
        return SplitTensor(y, groups={key: self.submodules[key].out_channels for key in self.keys})


def conv2d(x, w, complex):
    """ Real or complex convolution between x (B, C, M, N, [2]) and w (K, C, H, W, [2]), handles type problems.
    x and w can be real, complex, or real with an additional trailing dimension of size 2.
    A complex convolution causes the view or cast of both x and w as complex tensors.
    Returns a real or complex tensor of size (B, K, M', N'). """
    def real_to_complex(z):
        if z.is_complex():
            return z
        elif z.ndim == 5:
            # return torch.view_as_complex(z)  # View
            return torch.complex(z[..., 0], z[..., 1])  # Temporary copy instead of view...
        elif z.ndim == 4:
            return z.type(torch.complex64)  # Cast
        else:
            assert False

    if w.shape[0] == 0:  # Stupid special case because pytorch can't handle zero-sized convolutions.
        y = x[:, 0:0]  # (B, 0, M, N), this assumes that x is the right type
        if complex:
            y = real_to_complex(y)

    else:
        if complex:
            x = real_to_complex(x)
            w = real_to_complex(w)
            conv_fn = complex_conv2d
        else:
            conv_fn = F.conv2d
        y = conv_fn(x, w)

    return y
