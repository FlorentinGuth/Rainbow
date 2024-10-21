import argparse
import os
import random
import shutil
import time
import warnings
import sys

import numpy as np
import tensorflow  # might help with weird incompatibility error with protobuf? https://github.com/pytorch/pytorch/issues/81140
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from phase_scattering2d_torch import Scattering2D, ScatNonLinearity, ScatNonLinearityAndSkip, Realifier, Complexifier
from models.hidden_layer import hidden_layer
from models.Analysis import Analysis, AnalysisNonLinearity, StructuredAnalysis
from models.LinearProj import ComplexConv2d, TriangularComplexConv2d
from models.DCT import DCT
from models.STFT import STFT
from models.Classifier import Classifier
from models.LGM_logits import LGM_logits
from models.Standardization import Standardization, Normalization
from models.standard_custom import standard_models, LearnableNonLinearity
from datasets import get_dataloaders

from torch.utils.tensorboard import SummaryWriter
from utils import *


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--checkpoint-frequency', default=10, type=int, help='checkpoint the model in a separate file every such epochs. OBSOLETE, switched to 1/2/5eX hardcoded.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--loose-resume', action='store_true',
                        help='perform a non-strict resume (ignores missing/unexpected keys and shape errors)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    # Additional training args
    parser.add_argument('--learning-rate-adjust-frequency', default=30, type=int,
                        help='number of epoch after which learning rate is decayed by 10 (default: 30)')
    parser.add_argument('--dir', default='default_dir', type=str,
                        help='directory for training logs and checkpoints')
    parser.add_argument('--restart', help='when resuming training, start from epoch 0', action='store_true')
    parser.add_argument('--pars-reg', help='Parseval regularization', action='store_true')
    parser.add_argument('--quadrature-reg', help='Quadrature regularization', action='store_true')
    parser.add_argument('--beta', default=0.01, type=float, help='learning rate for pars reg')
    parser.add_argument('--gmm-lambda',  default=0., type=float, help='lambda for GMM likelihood loss')
    parser.add_argument('--lr-gmm-std', default=0.01, type=float, help='learning rate for gmm std')
    parser.add_argument('--stft-window', default='hanning', type=str, help='STFT window type')
    parser.add_argument('--stft-size', default=8, type=int, nargs='*', help='STFT window size')
    parser.add_argument('--stft-stride', default=4, type=int, nargs='*', help='STFT stride')
    parser.add_argument('--dct-type', default='II', type=str, help='DCT type')
    parser.add_argument('--dct-ortho', action='store_true', help='orthogonal DCT')
    parser.add_argument('--dct-size', default=8, type=int, nargs='*', help='DCT window size')
    parser.add_argument('--dct-stride', default=4, type=int, nargs='*', help='DCT stride')


    model_names = ['scatnetblockanalysis', 'scatnetblock', 'blockanalysis', 'stft', 'stftblock', 'stftblockanalysis',
                   'dct', 'dctblock', 'dctblockanalysis', 'hidden_layer']

    module_names = ["Fw", "B", "R", "C", "mod", "rho", "Std", "P", "Pr", "Pc", "N", "FrhoF", "STFT", "DCT", "HL", "id"]

    def parse_architecture(arch):
        """ Parses arch into the name of a standard architecture, or a list of modules which describe a block. """
        if arch in standard_models:
            return arch

        if arch in model_names:  # Predefined architectures
            block = []

            if 'scat' in arch:
                block.append('Fw')
                block.append('rho')
            elif 'stft' in arch:
                block.append('STFT')
                block.append('rho')
            elif 'dct' in arch:
                block.append('DCT')
                block.append('rho')
            elif arch == 'hidden_layer':
                block.append('HL')

            if 'block' in arch:
                block.append('Std')
                block.append('P')
                block.append('N')

            if 'analysis' in arch:
                block.append('FrhoF')
        else:
            block = arch.split()

        assert all(module in module_names for module in block)
        return block


    # Main pipeline arguments
    parser.add_argument('--n-blocks', type=int, default=1, help='number of blocks in the pipeline')
    parser.add_argument('-a', '--arch', type=parse_architecture,
                        help='model architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--first-arch', type=parse_architecture, default=[],
                        help='first architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--last-arch', type=parse_architecture, default=[],
                        help='last architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--tensor-blocks', action='store_true', help='blocks have tensors as inputs and outputs')
    parser.add_argument('--phi-arch', type=parse_architecture, default=[],
                        help='phi architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--psi-arch', type=parse_architecture, default=[],
                        help='psi architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--linear-arch', type=parse_architecture, default=[],
                        help='linear architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--non-linear-arch', type=parse_architecture, default=[],
                        help='non-linear architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--left-arch', type=parse_architecture, default=[],
                        help='left architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--right-arch', type=parse_architecture, default=[],
                        help='right architecture: ' + ' | '.join(model_names) + ' or description of a block')
    parser.add_argument('--complex', action='store_true', help='use complex coefficients (deprecated, use Pr/Pc)')
    parser.add_argument('--homogeneous', action='store_true', help='make the standardization layers homogeneous')


    # Scattering parameters
    parser.add_argument('--scattering-order2', type=int, choices=[0, 1], nargs='*',
                        help='compute order 2 scattering coefficients')
    parser.add_argument('--scattering-wph', type=int, choices=[0, 1], nargs='*', help='use phase scattering')
    parser.add_argument('--scat-angles', default=8, type=int, nargs='*', help='number of orientations for scattering')
    parser.add_argument('--scat-full-angles', type=int, choices=[0, 1], nargs='*', help='angles up to 2pi instead of pi')
    parser.add_argument('--scattering-J', default=1, type=int, nargs='*', help='maximum scale for the scattering transform')
    parser.add_argument('--scattering-scales-per-octave', default=1, type=int, help='number of scales per octave')
    parser.add_argument('--factorize-filters',  default=None, type=int, help='block number when we start factorizing'
                                                             'scattering filters in phi_1/2 - psi_1 / phi_1 - psi_3/2')
    parser.add_argument('--scat-non-linearity', default="mod", nargs='*',
                        help="non-linearity used after scattering (modules 'mod' and 'rho')")
    parser.add_argument('--scat-non-linearity-bias', default=None, nargs='*', type=eval,
                        help='bias/threshold for some scattering non-linearities')
    parser.add_argument('--scat-non-linearity-gain', default=None, nargs='*', type=eval,
                        help='gain for some scattering non-linearities')
    parser.add_argument('--scat-non-linearity-learned', type=int, choices=[0, 1], nargs='*',
                        help='learn the scattering non-linearity parameters')


    def parse_separation_sizes(s):
        """ Parses a semicolon-separated list of strings. """
        return tuple(e for e in s.split(';') if e != '')  # Can't eval because we then cannot json.dump the arguments


    # Linear projection parameters
    parser.add_argument('--L-proj-size', type=str, nargs='*', help='dimension of the linear projection')
    parser.add_argument('--Pr-size', type=str, nargs='*', help='dimension of the real linear projection')
    parser.add_argument('--Pc-size', type=str, nargs='*', help='dimension of the complex linear projection')
    parser.add_argument('--L-kernel-size', default=1, type=int, nargs='*', help='kernel size of L')
    parser.add_argument('--remove-L', type=int, choices=[0, 1], nargs='*', help='remove projection')
    parser.add_argument('--separate-orders', action='store_true', help='force (triangular) order separation')
    parser.add_argument('--diagonal-orders', action='store_true', help='diagonal order separation instead of triangular')
    parser.add_argument('--separate-freqs', action='store_true',
                        help='force frequency separation (translation equivariance)')
    parser.add_argument('--separate-angles', action='store_true', help='force angle separation (rotation equivariance)')
    parser.add_argument('--separate-packets', action='store_true', help='force linear operations on wavelet packets')
    parser.add_argument('--throw-packets', action='store_true', help='throw away linear part of F_w')
    parser.add_argument('--mix-input', help='1x1 transformation of input', action='store_true')
    parser.add_argument('--yuv', help='mix using YUV', action='store_true')
    parser.add_argument('--grayscale', help='use grayscale', action='store_true')
    parser.add_argument('--P-eigenvectors', default="", help='eigenvectors to constrain atoms in a subspace (None or "path")', nargs='*')
    parser.add_argument('--P-eigenvectors-start', default=None, type=int, help='first eigenvector to use', nargs='*')
    parser.add_argument('--P-eigenvectors-end', default=None, type=int, help='first eigenvector to drop', nargs='*')
    parser.add_argument('--P-eigenvalues', default="", help='eigenvalues to initialize with Gaussian distribution', nargs='*')
    parser.add_argument('--P-initialization', default="", nargs='*', help='paths for P initialization')
    parser.add_argument('--freeze-P', type=int, choices=[0, 1], nargs='*', help='freeze the weights of P during training')

    # Analysis parameters
    parser.add_argument('--dictionary-size', default=2048, type=int, nargs='*', help='size of the frame')
    parser.add_argument('--lambda-star', default=1., type=float, nargs='*', help='lambda_star')
    parser.add_argument('--dict-norm', default=0., type=float, nargs='*',
                        help='ratio max/min of dictionary norms allowed - 0 means no dictionary normalization')
    parser.add_argument('--load-proj', default='', type=str, help='Model from where to load proj after scat')
    parser.add_argument('--load-dict', default='', type=str, help='Model from where to load dict after scat')
    parser.add_argument('--non-linearity', default='relu', type=str, help='non linearity for analysis')
    parser.add_argument('--non-lin-delta', default=0.01, type=float,  nargs='*',
                        help='delta parameter for certain non linearities')
    parser.add_argument('--diagonal-analysis', action='store_true', help='diagonal Fj over the groups defined by Pj')
    parser.add_argument('--analysis-preserve-groups', type=int, nargs='*', help='groups to preserve in anayslis (identity)')

    # Classifier parameters
    parser.add_argument('--classifier-bias', action="store_const", default=True, const=True,
                        help="add a bias to the classifier (default)")
    parser.add_argument('--no-classifier-bias', action="store_false", dest="classifier_bias",
                        help="remove the bias to the classifier")
    parser.add_argument('--classifier-batch-norm', default="affine",
                        help="type of batch norm in the classifier (can be 'affine', 'std', or 'none')")
    parser.add_argument('--avg-ker-size', default=1, type=int, help='size of averaging kernel')
    parser.add_argument('--avgpool', help='Full avg pooling', action='store_true')
    parser.add_argument('--identity-classifier', action='store_true', help="force an identity classifier")
    parser.add_argument('--classifier-type', default='fc', type=str, help='classifier type')
    parser.add_argument('--gmm-std', help='Use std deviation in GMM', action='store_true')
    parser.add_argument('--gmm-alpha', default=0.1, type=float, help='margin in GMM')

    # Standard architecture parameters
    parser.add_argument('--standard-pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--standard-no-bias', action='store_true', help='remove bias from the architectures')
    parser.add_argument('--standard-classifier-no-bias', action='store_true', help='remove bias from the classifier')
    parser.add_argument('--standard-batch-norm', default="pre", choices=["none", "pre", "post"],
                        help="can be 'none' (no batch norms), 'pre' (before non-linearity), 'post' (after non-linearity)")
    parser.add_argument('--standard-width-scaling', type=float, default=1, help='width multiplier in all layers')
    parser.add_argument('--standard-non-linearity', default="relu", nargs="+", help="non-linearity to use")
    parser.add_argument('--standard-init-gain', default=1.0, type=float,
                        help="initialization of gain parameter of non-linearities")
    parser.add_argument('--standard-init-bias', default=0.0, type=float,
                        help="initialization of bias parameter of non-linearities")

    # Dataset parameter
    parser.add_argument('--data', metavar='DIR', help='path to dataset (ImageNet only)')
    parser.add_argument('--dataset', default="ImageNet", help="dataset to use")
    parser.add_argument('--cifar10', action='store_const', const="CIFAR10", dest="dataset", help='use CIFAR10 dataset')
    parser.add_argument('--cifar100', action='store_const', const="CIFAR100", dest="dataset",
                        help='use CIFAR100 dataset')
    parser.add_argument('--mnist', action='store_const', const="MNIST", dest="dataset", help='use MNIST dataset')
    parser.add_argument('--data-subset', type=eval, default=None,
                        help='expression (typically a range) to select a subset of the training set')
    parser.add_argument('--classes-subset', type=int, nargs='*',
                        help='select a subset of classes for training and evaluation')
    parser.add_argument('--resize-images', default=None, type=int, help='resize images to this resolution')
    parser.add_argument('--randomize-labels', metavar='SEED', default=None, type=int,
                        help='randomize labels with a given seed')

    return parser


parser = build_parser()


def main():
    args = get_args()
    main_worker(args)


def get_args(*args_to_parser):
    args = parser.parse_args(*args_to_parser)
    job_id_field = "SLURM_ARRAY_TASK_ID"
    if job_id_field in os.environ:
        job_id = int(os.environ[job_id_field])
        args.dir = f"{args.dir}-init{job_id:02}"

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    n_blocks = args.n_blocks

    # Other int variables need to be specified

    for item in ["scattering_order2", "scattering_wph", "remove_L", "scat_non_linearity_learned", "freeze_P"]:
        if getattr(args, item) is None:
            setattr(args, item, [False] * n_blocks)
        elif len(getattr(args, item)) < n_blocks:
            setattr(args, item, getattr(args, item) + [False] * (n_blocks - len(getattr(args, item))))

    for item in ["scat_angles", "scat_full_angles", "scattering_J",
                 "scat_non_linearity", "scat_non_linearity_gain", "scat_non_linearity_bias",
                 "stft_size", "stft_stride", "dct_size", "dct_stride",
                 "dictionary_size", "dict_norm", "lambda_star", "non_lin_delta",
                 "L_proj_size", "L_kernel_size", "P_eigenvectors", "P_eigenvectors_start", "P_eigenvectors_end",
                 "P_eigenvalues", "P_initialization"]:
        if type(getattr(args, item)) != list:  # value repeated
            setattr(args, item, [getattr(args, item)] * n_blocks)
        elif len(getattr(args, item)) < n_blocks:  # default value added
            setattr(args, item, getattr(args, item) +
                    [parser.get_default(item)] * (n_blocks - len(getattr(args, item))))

    return args


logfiles = []


def new_logfile(path):
    """ Returns the logfile, which is an open file (needs to be closed). """
    logfile = open(path, 'a')
    logfiles.append(logfile)
    return logfile


def build_layers(input_type: SplitTensorType, modules, i, args):
    """ Builds a list of layers from a list of modules.
    :param input_type:
    :param modules: list of strings describing the layers
    :param i: index of the current block
    :param args: command-line arguments
    :return: Sequential module
    """
    builder = Builder(input_type)

    def branching_kwargs(**submodules):
        """ Builds kwargs for Branching. Expects a dict of module_name -> architecture list of strings. """
        kwargs = {}
        for name, arch in submodules.items():
            kwargs[f"{name}_module_class"] = build_layers
            kwargs[f"{name}_module_kwargs"] = dict(modules=arch, i=i, args=args)
        return kwargs

    for module in modules:
        if module == "Fw":
            kwargs = dict(
                scales_per_octave=args.scattering_scales_per_octave,
                L=args.scat_angles[i], full_angles=args.scat_full_angles[i], separate_freqs=args.separate_freqs,
            )
            if args.factorize_filters is not None:
                if i < args.factorize_filters:
                    kwargs.update(scales_per_octave=1)
                else:
                    kwargs.update(factorize_filters=True, i=(i - args.factorize_filters) % 2)

            if args.scattering_wph[i]:
                builder.add_layer(Scattering2D, kwargs)
                kwargs = dict(phi=args.phi_arch)
                if args.scat_angles[i] > 0:
                    kwargs["psi"] = args.psi_arch
                builder.add_layer(Branching, branching_kwargs(**kwargs))
            else:
                assert False

        elif module == "R":
            builder.add_batched(Realifier)

        elif module == "C":
            builder.add_batched(Complexifier)

        elif module == "B":
            builder.add_layer(Forker, dict(group_names=["left", "right"]))
            builder.add_layer(Branching, branching_kwargs(left=args.left_arch, right=args.right_arch))

        elif module in ["mod", "rho"]:
            kwargs = dict(
                non_linearity=args.scat_non_linearity[i], separate_orders=args.separate_orders,
                bias=args.scat_non_linearity_bias[i], gain=args.scat_non_linearity_gain[i],
                learned_params=args.scat_non_linearity_learned[i],
            )
            if module == "mod":
                builder.add_layer(ScatNonLinearity, kwargs)
            else:  # module == "rho"
                builder.add_layer(ScatNonLinearityAndSkip, kwargs)
                builder.add_layer(Branching, branching_kwargs(linear=args.linear_arch, non_linear=args.non_linear_arch))

        elif module == "Std":
            builder.add_batched(Standardization, dict(remove_mean=not args.homogeneous))

        elif module in ["P", "Pr", "Pc"]:
            if args.remove_L[i]:
                pass
            else:
                out_channels_args = dict(P=args.L_proj_size, Pr=args.Pr_size, Pc=args.Pc_size)[module]
                out_channels = eval(out_channels_args[i])
                # Convert to dictionary (providing size for each group).
                if not isinstance(out_channels, dict):
                    # Order separation.
                    if args.separate_orders:
                        if isinstance(out_channels, tuple):  # (total_size, p)
                            max_order = max((key[1] for key in builder.input_type.groups.keys()))
                            out_channels = binom_to_sizes(max_order, *out_channels)
                        assert isinstance(out_channels, list) or isinstance(out_channels, np.ndarray)
                        out_channels = {i: out_channels[i] for i in range(len(out_channels))}
                    else:
                        out_channels = {0: out_channels}  # No order separation
                    # Now out_channels is a dict from order to group sizes.

                # Add frequency in keys, case it is missing.
                if isinstance(next(out_channels.__iter__()), int):
                    # Freq separation: split each group across frequencies equally.
                    if args.separate_freqs:
                        num_freqs = 1 + args.scattering_scales_per_octave * args.scat_angles[i]
                    else:
                        num_freqs = 1

                    # Divide the number of output channels for each frequency, taking care of identity case.
                    out_channels = {(freq, order): ceil_div(size, num_freqs) if isinstance(size, int) else size
                                    for order, size in out_channels.items() for freq in range(num_freqs)}

                # Determine type of weights (default is type of input).
                complex_weights = dict(P=None, Pr=False, Pc=True)[module]

                kwargs = dict(
                    complex_weights=complex_weights, out_channels=out_channels, kernel_size=args.L_kernel_size[i],
                    parseval=args.pars_reg, quadrature=args.quadrature_reg, eigenvectors=args.P_eigenvectors[i],
                    eigenvectors_start=args.P_eigenvectors_start[i], eigenvectors_end=args.P_eigenvectors_end[i],
                    eigenvalues=args.P_eigenvalues[i], initialization=args.P_initialization[i],
                )

                if args.separate_orders and (not args.diagonal_orders):
                    if args.separate_freqs:
                        raise ValueError("Frequency separation and triangular order separation not implemented")
                    else:
                        layer = builder.add_layer(TriangularComplexConv2d, kwargs)
                else:
                    # Diagonal over frequencies (if separated) and orders (if separated)
                    layer = builder.add_diagonal(ComplexConv2d, kwargs)

                if args.freeze_P[i]:
                    layer.apply(lambda m: set_requires_grad(m, False))

        elif module == "N":
            builder.add_diagonal(Normalization)

        elif module == "FrhoF":
            # TODO: this is not the right lambda, and distribute the frame size everywhere.
            non_lin = AnalysisNonLinearity(
                args.non_linearity, lambd=args.lambda_star[i] / np.sqrt(args.dictionary_size[i]),
                delta=args.non_lin_delta[i],
            )

            kwargs = dict(non_linearity=non_lin, norm_ratio=args.dict_norm[i], parseval=args.pars_reg,
                          quadrature=args.quadrature_reg)

            if args.diagonal_analysis:
                assert False  # deprecated.
                analysis = StructuredAnalysis(
                    in_channels=groups, frame_total_size=args.dictionary_size[i],
                    preserve_groups=args.analysis_preserve_groups, **kwargs)
            else:
                builder.add_diagonal(Analysis, dict(frame_size=args.dictionary_size[i], **kwargs))

        elif module == "STFT":
            assert False
            stft = STFT(args.stft_size[i], args.stft_window, args.stft_stride[i], args.complex)
            if args.stft_window in ['hanning', 'gaussian']:
                n_space = n_space // args.stft_stride[i] + 1
            else:  # rectangle
                n_space = (n_space - args.stft_window) // args.stft_stride[i] + 1
            old_nb_channels_in = nb_channels_in
            nb_channels_in *= (2 - args.complex) * args.stft_size[i] ** 2 - (1 - args.complex)

            if args.scattering_phase_channels == 'relu':
                groups = {0: 2 * nb_channels_in}
            else:
                if args.no_preserve_low_freq:
                    groups = {0: old_nb_channels_in, 1: nb_channels_in - old_nb_channels_in, 2:
                        nb_channels_in}
                else:
                    groups = {0: old_nb_channels_in, 1: nb_channels_in - old_nb_channels_in, 2:
                        nb_channels_in - old_nb_channels_in}

            layers.append(stft)

        elif module == "DCT":
            assert False
            dct = DCT(args.dct_size[i], args.dct_type, args.dct_stride[i], args.dct_ortho)
            n_space = n_space // args.dct_stride[i] + 1
            old_nb_channels_in = nb_channels_in
            nb_channels_in *= args.dct_size[i] ** 2

            if args.scattering_phase_channels == 'relu':
                groups = {0: 2 * nb_channels_in}
            else:
                if args.no_preserve_low_freq:
                    groups = {0: old_nb_channels_in, 1: nb_channels_in - old_nb_channels_in, 2:
                        nb_channels_in}
                else:
                    groups = {0: old_nb_channels_in, 1: nb_channels_in - old_nb_channels_in, 2:
                        nb_channels_in - old_nb_channels_in}

            layers.append(dct)

        elif module == "HL":
            assert False
            hidden = hidden_layer(args.dictionary_size[i], args.dict_kernel_size[i], args.dict_stride[i],
                                  nb_channels_in)
            nb_channels_in = args.dictionary_size[i]
            n_space = (n_space - args.dict_kernel_size[i]) // args.dict_stride[i] + 1
            layers.append(hidden)

        elif module == "id":
            builder.add_layer(Identity)

        else:
            assert False

    if modules == args.arch and args.tensor_blocks:
        groups = builder.input_type.groups
        builder.add_layer(ToTensor)
        builder.add_layer(ToSplitTensor, dict(groups=groups))

    return builder.module()


def load_model(args, logfile, summaryfile, writer, log=True):
    if isinstance(args.arch, str):
        # Standard architecture.
        if args.standard_pretrained:
            model = standard_models[args.arch](pretrained=True)
        else:
            model = standard_models[args.arch](
                non_linearity=LearnableNonLinearity, non_linearity_name=args.standard_non_linearity,
                batch_norm=args.standard_batch_norm, width_scaling=args.standard_width_scaling,
                bias=not args.standard_no_bias, classifier_bias=not args.standard_classifier_no_bias,
                init_gain=args.standard_init_gain, init_bias=args.standard_init_bias,
            )

    else:
        # Learned Scattering architecture, arch is the description of a block.

        if args.dataset == "MNIST":
            nb_channels_in = 1
        else:
            nb_channels_in = 3
        if args.grayscale:
            nb_channels_in = 1

        if args.dataset == "ImageNet":
            n_space = 224
        elif args.dataset == "MNIST":
            n_space = 28
        else:
            n_space = 32
        if args.resize_images is not None:
            n_space = args.resize_images

        if args.dataset.startswith("ImageNet"):
            num_classes = 1000
        elif args.dataset == "CIFAR100":
            num_classes = 100
        else:
            num_classes = 10

        input_type = TensorType(num_channels=nb_channels_in, spatial_shape=(n_space, n_space), complex=False)
        builder = Builder(input_type)

        if args.mix_input:
            builder.add_layer(ComplexConv2d, dict(out_channels=nb_channels_in, kernel_size=1,
                                                  parseval=args.pars_reg, quadrature=args.quadrature_reg))

            if args.yuv:
                yuv_weight = torch.FloatTensor(
                    [[0.299, 0.587, 0.114], [-0.147, - 0.289, 0.436], [0.615, -0.515, -0.100]]).reshape(3, 3, 1, 1)
                if args.complex:
                    yuv_weight = yuv_weight.type(torch.complex64)
                    yuv_weight = torch.view_as_real(yuv_weight)
                builder.layers[-1].param.data = yuv_weight
                builder.layers[-1].param.requires_grad_(False)
                builder.layers[-1].parseval = False
                builder.layers[-1].quadrature = False

        builder.add_layer(ToSplitTensor, dict(groups={(0, 0): nb_channels_in}))
        for i in range(args.n_blocks):
            if i == 0 and args.first_arch != []:
                arch = args.first_arch
            elif i == args.n_blocks - 1 and args.last_arch != []:
                arch = args.last_arch
            else:
                arch = args.arch
            builder.add_layer(build_layers, dict(modules=arch, i=i, args=args))
        builder.add_layer(ToTensor)

        if args.classifier_type == 'gmm':
            assert False
            classifier = LGM_logits(n_space, nb_channels_in, alpha=args.gmm_alpha, nb_classes=num_classes,
                                    use_std=args.gmm_std, avg_ker_size=args.avg_ker_size, avgpool=args.avgpool,
                                    bottleneck=args.bottleneck, bottleneck_size=args.bottleneck_size)
        else:
            builder.add_layer(
                Classifier, dict(
                    nb_classes=num_classes, avg_ker_size=args.avg_ker_size, avgpool=args.avgpool,
                    identity=args.identity_classifier,
                    bias=args.classifier_bias, batch_norm=args.classifier_batch_norm,
                ),
            )

        model = builder.module()

    if log:
        print_and_write(str(model), logfile, summaryfile)
        num_params = num_parameters(model, logfile, summaryfile, log=True)
        if writer is not None:
            writer.add_scalar('num_params', num_params, global_step=0)
        print_and_write('Number of epochs {}, learning rate decay epochs {}'.format(
            args.epochs, args.learning_rate_adjust_frequency), logfile, summaryfile)

    return model


def main_worker(args):
    best_acc1 = 0
    best_acc5 = 0
    best_epoch_acc1 = 0
    best_epoch_acc5 = 0

    n_blocks = args.n_blocks

    file_suffix = f"batchsize_{args.batch_size}_lrfreq_{args.learning_rate_adjust_frequency}"

    checkpoint_savedir = os.path.join('./checkpoints', args.dir)
    if not os.path.exists(checkpoint_savedir):
        os.makedirs(checkpoint_savedir)
    checkpoint_savefile = os.path.join(checkpoint_savedir, f'{file_suffix}.pth.tar')

    logs_dir = os.path.join('./training_logs', args.dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logfile = new_logfile(os.path.join(logs_dir, f'{file_suffix}.log'))
    summaryfile = new_logfile(os.path.join(logs_dir, 'summary_file.txt'))

    writer = SummaryWriter(logs_dir)

    # Also save args.
    with open(os.path.join(checkpoint_savedir, "args.json"), 'w') as f:
        import json
        json.dump(args.__dict__, f, indent=2, default=str)
    print_and_write(f"Command line: {' '.join(sys.argv)}", logfile, summaryfile)

    train_loader, val_loader = get_dataloaders(args, logfile, summaryfile)

    model = load_model(args, logfile, summaryfile, writer)

    model = torch.nn.DataParallel(model)
    model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    assert args.classifier_type != 'gmm'  # Deprecated here, params should be dealt with in a special way?
    if args.pars_reg or args.quadrature_reg:
        # Do not put any weight decay for pars or quadrature reg weights within the optimizer.
        special_params, all_other_params = [], []
        for module in model.modules():  # Iterate over all submodules
            # Immediate parameters which require gradients of this module.
            parameters = [param for param in module.parameters(recurse=False) if param.requires_grad]
            if getattr(module, "parseval", False) or getattr(module, "quadrature", False):  # Is this a parseval or quadrature module?
                special_params.extend(parameters)
            else:
                all_other_params.extend(parameters)

        optimizer = torch.optim.SGD(
            [{'params': all_other_params}, {'params': special_params, 'weight_decay': 0.}],
            args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    if args.load_proj:
        if os.path.isfile(args.load_proj):
            print_and_write("=> Loading linear projs from checkpoint '{}'".format(args.load_proj), logfile)
            checkpoint = torch.load(args.load_proj)
            checkpoint_dict = checkpoint['state_dict']
            for i in range(len(args.separate_orders)):
                projs = model[i].module.linear_proj.proj.projs
                for l in range(len(projs)):
                    projs[l].data = checkpoint_dict['{}.module.linear_proj.proj.projs.{}'.format(i,l)].data
                    projs[l].requires_grad = False

            for i in range(len(args.separate_orders), n_blocks):
                model[i].module.linear_proj.proj.data = checkpoint_dict['{}.module.linear_proj.proj'.format(i)].data
                model[i].module.linear_proj.proj.requires_grad = False
            print_and_write(
                "=> loaded linear projs from checkpoint '{}' (epoch {})".format(args.load_proj, checkpoint['epoch']),
                logfile)
        else:
            print_and_write("=> no checkpoint found at '{}'".format(args.load_proj), logfile)
            return

    if args.load_dict:
        if os.path.isfile(args.load_dict):
            print_and_write("=> Loading dictionaries from checkpoint '{}'".format(args.load_dict), logfile)
            checkpoint = torch.load(args.load_dict)
            checkpoint_dict = checkpoint['state_dict']
            for i in range(n_blocks):
                model[i].module.analysis.dictionary_weight.data = checkpoint_dict['{}.module.analysis.dictionary_weight'.format(i)].data
                model[i].module.analysis.apply(lambda m: set_requires_grad(m, False))
            print_and_write(
                "=> loaded dicts from checkpoint '{}' (epoch {})".format(args.load_dict, checkpoint['epoch']),
                logfile)
        else:
            print_and_write("=> no checkpoint found at '{}'".format(args.load_dict), logfile)
            return

    if args.resume:
        if os.path.isfile(args.resume):
            print_and_write("=> loading checkpoint '{}'".format(args.resume), logfile, summaryfile)
            checkpoint = torch.load(args.resume)
            if not args.restart:
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
            model_state_dict = checkpoint['state_dict']
            model_dict = model.module.state_dict()
            new_model_state_dict = {}
            for key, val in model_state_dict.items():
                if "lambda_star" in key:
                    continue
                if "analysis" in key:
                    before, after = key.split("analysis")
                    if not after[1].isdigit():  # Old checkpoint, before analysis is Sequential
                        after = f".0{after}"
                    key = f"{before}analysis{after}"
                new_model_state_dict[key] = val
            model_dict.update(new_model_state_dict)

            try:
                model.module.load_state_dict(model_dict)
            except RuntimeError as err:
                if args.loose_resume:
                    print_and_write(f"Loose resume, ignored errors: {err}", logfile, summaryfile)
                else:
                    raise err

            if not args.restart:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print_and_write("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), logfile,
                            summaryfile)
        else:
            print_and_write("=> no checkpoint found at '{}'".format(args.resume), logfile, summaryfile)

    cudnn.benchmark = True

    print_model_info(model, logfile, summaryfile)

    if args.evaluate:
        print_and_write("Evaluating model at epoch {}...".format(args.start_epoch), logfile)
        one_epoch(
            loader=val_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=args.start_epoch, args=args,
            logfile=logfile, summaryfile=summaryfile, writer=writer, is_training=False,
        )
        return

    # Weird loop logic, so that epoch is the number of training epochs done (counting the current one).
    # We do one validation epoch + checkpointing at initialization as well.
    epoch = args.start_epoch
    while True:
        # evaluate on validation set
        acc1, acc5 = one_epoch(
            loader=val_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, args=args,
            logfile=logfile, summaryfile=summaryfile, writer=writer, is_training=False,
        )

        # Remember best acc@1 and save checkpoint.
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch_acc1 = epoch
        if acc5 > best_acc5:
            best_acc5 = acc5
            best_epoch_acc5 = epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_acc1': best_acc1,
            # 'optimizer': optimizer.state_dict(),  # Save space at the cost of reproducibility (don't save gradient momenta).
        }, is_best, checkpoint_filename=checkpoint_savefile, epoch=epoch)

        # Stop if we are at the last epoch.
        if epoch == args.epochs:
            break

        # Prepare for the training epoch (epoch is the number of completed training epochs).
        adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch (now epoch counts the current training epoch).
        epoch += 1
        one_epoch(
            loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, args=args,
            logfile=logfile, summaryfile=summaryfile, writer=writer, is_training=True,
        )

    print_model_info(model, logfile, summaryfile)

    print_and_write(
        "Best top 1 accuracy {:.2f} at epoch {}, best top 5 accuracy {:.2f} at epoch {}".format(
            best_acc1, best_epoch_acc1, best_acc5, best_epoch_acc5), logfile, summaryfile)


@torch.no_grad()
def print_model_info(model, logfile, summaryfile):
    model_info = ["Model info:"]

    for module_name, module in model.named_modules():
        module_info = []

        # Submodule-specific extra info
        if hasattr(module, "model_info"):
            module_info.extend(module.model_info())

        for frame_name, frame in parseval_frames(module):
            if frame.shape[0] > frame.shape[1]:
                gram = frame.conj().t() @ frame
            else:
                gram = frame @ frame.conj().t()
            singular_values = torch.symeig(gram)[0]
            min_sv, max_sv = singular_values[0], singular_values[-1]
            module_info.append(
                f"\n  - Parseval on {frame_name}: singular values {tensor_summary_stats(singular_values)}")

        for frame_name, frame in quadrature_frames(module):
            #norm_ratio = torch.norm(frame.t() @ frame)/(torch.norm(frame)**2)
            norm_ratio = torch.norm(frame @ frame.t()) / (torch.norm(frame)**2)
            norm_frame = torch.norm(frame)
            #module_info.append(f"\n  - Quadrature on {frame_name}: ratio norm W^T W / norm W^2 {norm_ratio:.3f}")
            module_info.append(f"\n  - Quadrature on {frame_name}: ratio norm W W^T / norm W^2 {norm_ratio:.3f}")
            module_info.append(f"\n  - Frame norm of {frame_name}: {norm_frame:.3f}")

        if len(module_info) > 0:
            model_info.extend([f"\n- {module_name} ({module.__class__.__name__}): "] + module_info)

    print_and_write("".join(model_info), logfile, summaryfile)


def one_epoch(loader, model, criterion, optimizer, epoch, args, logfile, summaryfile, writer, is_training):
    batch_time = AverageMeter('Time', ':.1f')
    data_time = AverageMeter('Data', ':.1f')
    loss = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.1f')
    top5 = AverageMeter('Acc@5', ':.1f')
    name_epoch = "Train" if is_training else "Validation"
    progress = ProgressMeter(
        len(loader), [batch_time, data_time, loss, top1, top5],
        prefix="{} Epoch: [{}]".format(name_epoch, epoch))

    if is_training:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_training):
        end = time.time()
        for i, (input, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            input = model(input)
            if args.classifier_type == 'gmm':
                output, likelihood_loss = input
            else:
                output = input

            loss_batch = criterion(output, target)
            if is_training and args.classifier_type == 'gmm' and args.gmm_lambda > 0:
                loss_batch += args.gmm_lambda * (likelihood_loss.mean())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss.update(loss_batch.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            if is_training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

                # Parseval step
                on_weight_update(model, args)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_and_write('\n', logfile)
                progress.display(i, logfile)

    # Print statistics summary
    logfiles = [logfile, summaryfile]
    if not is_training and epoch == 1:
        epoch_text = ' (First epoch)'
    elif not is_training and epoch == args.epochs:
        epoch_text = ' (Final epoch)'
    elif not is_training and epoch > 0 and (epoch % args.learning_rate_adjust_frequency) == 0:
        epoch_text = ' (before learning rate adjustment n° {})'.format(epoch // args.learning_rate_adjust_frequency)
    else:
        epoch_text = ''
        logfiles = [logfile, None]
    print_and_write('\n{} Epoch {}{}, * Acc@1 {:.2f} Acc@5 {:.2f}'.
                    format(name_epoch, epoch, epoch_text, top1.avg, top5.avg), *logfiles)
    print_model_info(model, logfile, summaryfile=None)

    if writer is not None:
        suffix = "train" if is_training else "val"
        writer.add_scalar(f"loss_{suffix}", loss.avg, global_step=epoch)
        writer.add_scalar(f"top5_{suffix}", top5.avg, global_step=epoch)
        writer.add_scalar(f"top1_{suffix}", top1.avg, global_step=epoch)

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, epoch, checkpoint_filename):
    torch.save(state, checkpoint_filename)
    # Save on epochs 0, 1, 2, 5, 10, 20, 50, 100, 200, 500...
    # Factor is the lowest power of 10 less than or equal to epoch.
    if epoch == 0 or (epoch % (factor := 10 ** int(np.log10(epoch))) == 0 and epoch // factor in [1, 2, 5]):
        shutil.copyfile(checkpoint_filename, checkpoint_filename.replace(".pth.tar", f"_{epoch}.pth.tar"))
    if is_best:
        shutil.copyfile(checkpoint_filename, checkpoint_filename.replace(".pth.tar", "_best.pth.tar"))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.learning_rate_adjust_frequency))
    if args.classifier_type == 'gmm' and args.gmm_std:
        lr_gmm_std = args.lr_gmm_std * (0.1 ** (epoch // args.learning_rate_adjust_frequency))

    for param_group in optimizer.param_groups:
        if 'name' in param_group.keys() and param_group['name'] == 'lr_std':
            param_group['lr'] = lr_gmm_std
        else:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_lambda_0(loader, model, n_dict_blocks, nb_batches=1):
    with torch.no_grad():

        best_lambda = torch.zeros(n_dict_blocks).cuda()

        for i, (input, target) in enumerate(loader):
            if i >= nb_batches:
                break
            input = input.cuda()
            for j in range(n_dict_blocks):
                input, lambda_max_batch = model[j](input)[:2]
                if lambda_max_batch.mean() > best_lambda[j]:
                    best_lambda[j] = lambda_max_batch.mean()

        return best_lambda


def set_requires_grad(m, requires_grad):
    for param in m.parameters():
        param.requires_grad_(requires_grad)


def num_parameters(model, logfile, summaryfile, log=True):
    total = 0
    s = ["Model parameters:\n"]
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num = param.numel()
        total += num
        s.append(f"- {name}: {param.shape} i.e. {num:n} parameters\n")
    s.append(f"Total: {total:n} parameters")
    if log:
        print_and_write(" ".join(s), logfile, summaryfile)
    return total


@torch.no_grad()
def on_weight_update(model, args):
    """" To call after every weight update (gradient step), to apply Parseval step and frame normalization. """
    beta = args.beta

    for module_name, module in model.named_modules():
        # Submodule-specific updates
        if hasattr(module, "on_weight_update"):
            module.on_weight_update()

        for _, frame in parseval_frames(module, update=True):
            N, C = frame.shape  # N: number of atoms, C input dimension

            # F = (1 + beta) F - beta F F^H F ; F F^H is (N, N), F^T F is (C, C)
            if N > C:  # typical case, better to compute F^H F (C, C)
                prod = torch.matmul(frame, torch.matmul(frame.conj().t(), frame))
            else:  # better to compute F F^H (N, N)
                prod = torch.matmul(torch.matmul(frame, frame.conj().t()), frame)

            frame.copy_((1 + beta) * frame - beta * prod)

        for _, frame in quadrature_frames(module, update=True):
            N, C = frame.shape  # N: number of atoms, C input dimension

            # F = F - beta F^* F^T F ; F^* F^T is (N, N), F^T F is (C, C)
            if N > C:  # typical case, better to compute F^T F (C, C)
                #prod = torch.matmul(frame.conj(), torch.matmul(frame.t(), frame))
                prod = torch.matmul(frame, torch.matmul(frame.t(), frame.conj()))
            else:  # better to compute F^* F^T (N, N)
                #prod = torch.matmul(torch.matmul(frame.conj(), frame.t()), frame)
                prod = torch.matmul(torch.matmul(frame, frame.t()), frame.conj())

            frame.copy_(frame - beta * prod)


def parseval_frames(module, update=False):
    """ Yield name, frame pairs. frame is of shape (N, C) and can be complex.
    If update is True, the caller will do an in-place update of the frames, so there should not be copies. """
    if getattr(module, "parseval", False):
        # This is a Parseval module.

        if hasattr(module, "full_weights"):
            # Bypass parameters, yield only the full weights for Parseval regularization.
            #if update:
            #    raise ValueError("In-place update of modules with `full_weights` method"
            #                     " (such as TriangularConv) is not implemented")
            frames = [("full_weight", module.full_weights())]
        else:
            # Parseval frames are immediate parameters which require grad.
            frames = [(name, param) for name, param in module.named_parameters(recurse=False) if param.requires_grad]

        complex = getattr(module, "complex_weights", False)
        for frame_name, frame_param in frames:
            frame = frame_param.data
            if complex:
                frame = torch.view_as_complex(frame)  # Cast back the frame to complex.
            frame = frame.reshape((frame_param.shape[0], -1))

            yield frame_name, frame


def quadrature_frames(module, update=False):
    """ Yield name, frame pairs. frame is of shape (N, C) and can be complex. """
    if getattr(module, "quadrature", False):
        # This is a quadrature module.

        if hasattr(module, "full_weights"):
            # Bypass parameters, yield only the full weights for quadrature regularization.
            #if update:
            #    raise ValueError("In-place update of modules with `full_weights` method"
            #                     " (such as TriangularConv) is not implemented")
            frames = [("full_weight", module.full_weights())]
        else:
            # Quadrature frames are immediate parameters which require grad.
            frames = [(name, param) for name, param in module.named_parameters(recurse=False) if param.requires_grad]

        complex = getattr(module, "complex_weights", False)
        for frame_name, frame_param in frames:
            frame = frame_param.data
            if complex:
                frame = torch.view_as_complex(frame)  # Cast back the frame to complex.
            frame = frame.reshape((frame_param.shape[0], -1))

            yield frame_name, frame


if __name__ == '__main__':
    main()
