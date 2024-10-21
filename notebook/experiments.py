""" Provides facilities for loading several experiments, and caching stuff related to them. """

import argparse
import json
from pathlib import Path
from functools import cached_property, lru_cache
from typing import *
import re
import warnings
import hashlib

import numpy as np
import torch

from . import models
from . import linalg
import main_block
import datasets
from utils import tensor_summary_stats, print_and_write


tqdm_enabled = True

def tqdm(it, *args, **kwargs):
    if tqdm_enabled:
        from tqdm.notebook import tqdm as t
        return t(it, *args, **kwargs)
    else:
        return it


global_checkpoints_dir = Path("checkpoints")


def short_hash(string, length=8):
    """ Return a short hash of a string, for shortening file names used for memoization. """
    h = hashlib.md5()
    h.update(string.encode())
    return h.hexdigest()[-length:]


def to_device(x, cuda):
    if isinstance(x, dict):
        return {k: to_device(v, cuda) for k, v in x.items()}
    elif isinstance(x, list):
        return [to_device(v, cuda) for v in x]
    elif isinstance(x, tuple):
        return tuple(to_device(v, cuda) for v in x)
    else:
        return x.to(device="cuda" if cuda else "cpu")


def memory_memoize(cache, name, closure, recompute=False, log=False, cuda=True):
    """ Memoizes the result of closure in the provided cache.
    :param cache: dictionary, name -> result
    :param name: name associated to given closure
    :param closure: callable with no parameters, returns result
    :param recompute: whether to recompute the result even if it is in cache
    :param log: whether to print cache misses
    :return: result of closure
    """
    if recompute or name not in cache:
        if log:
            print_and_write(f"Cache miss for {name}", exps.logfile)
        cache[name] = closure()
    return to_device(cache[name], cuda)
#     return bulk_memory_memoize(cache, {name: None}, lambda _: {name: closure()}, recompute=recompute, log=log)[name]


def bulk_memory_memoize(cache, names, closure, recompute=False, log=False):
    """ Memoizes the results of a bulk closure in the provided cache.
    Calls closure on a dictionary of keys that are to be computed.
    :param cache: dictionary, name -> result
    :param names: dictionary, name -> key
    :param closure: callable with parameters a dictionary of name -> key, returns dictionary of name -> result
    :param recompute: whether to recompute the result even if it is in cache
    :param log: whether to print cache misses
    :return: dictionary of name -> memoize
    """
    misses = {}
    for name, key in names.items():
        if recompute or name not in cache:
            misses[name] = key
    if len(misses) > 0:
        if log:
            print_and_write(f"Cache misses: {misses}", exps.logfile)
        for name, result in closure(misses).items():
            cache[name] = result
    return {name: cache[name] for name in names.keys()}


def disk_memoize(path, closure, recompute=False, log=False, cuda=True):
    """ Memoizes the result of closure on disk.
    :param path: path to the file used for saving the result
    :param closure: callable with no parameters, returns result
    :param recompute: whether to recompute the result even if it is on disk
    :param log: whether to print loads/stores
    :return: result of closure
    """
    if not isinstance(path, Path):
        path = Path(path)
    if (not recompute) and path.exists():
        if log:
            print_and_write(f"Load {path}", exps.logfile)
        result = torch.load(str(path), map_location="cuda" if cuda else "cpu")
    else:
        if log:
            print_and_write(f"Compute and store {path}", exps.logfile)
        result = to_device(closure(), cuda)
        torch.save(result, str(path))
    return result
#     return bulk_disk_memoize(cache, {name: None}, lambda _: {name: closure()}, recompute=recompute, log=log)[name]


def bulk_disk_memoize(paths, closure, recompute=False, log=False):
    """ Memoizes the results of a bulk closure on disk.
    :param paths: dictionary, path -> path to the file used for saving the result
    :param closure: callable with no parameters, returns result
    :param recompute: whether to recompute the result even if it is on disk
    :param log: whether to print loads/stores
    :return: result of closure
    """
    if not isinstance(path, Path):
        path = Path(path)
    if (not recompute) and path.exists():
        if log:
            print_and_write(f"Load {path}", exps.logfile)
        result = torch.load(str(path))
    else:
        if log:
            print_and_write(f"Compute and store {path}", exps.logfile)
        result = closure()
        torch.save(result, str(path))
    return result


def path_to_name(path: Union[str, Path]) -> str:
    return str(path).replace('/', '-')  # Alternative to path for naming purposes. Can be modified externally.


class Experiment:
    """ An Experiment holds the results of a previously-ran experiment. """
    def __init__(self, exps: "Experiments", path: Union[str, Path]):
        """ path is the path to suffix to global_checkpoints_dir (the experiment name with an optional jzarka/). """
        self.exps = exps
        self.path = path
        self.name = path_to_name(path)

        # Old-fashioned pairwise alignment. Deprecated?
        self.data_covariances = {}  # dataset_key, epoch, other_exp_name other_epoch, j -> cross covariance with other experiment (DecomposedMatrix)

        self.default_average_key = None
        self.mean_weight_covariances = {}  # average_key, j -> covariance of weights of mean representation (DecomposedMatrix)
        self.mean_alignments = {}  # average_key, j -> alignment with mean of shape (D_exp, D_mean)

        self.cache = {}  # key -> cached result

    def clear(self):
        """ Empties cache to free memory. """
        self.cache.clear()

    def memoize(self, name, closure, in_mem=True, on_disk=True, recompute=False, log=False, cuda=True):
        """ Memoizes a computation in memory (self.cache) and/or on disk (self.checkpoints_dir). """
        if on_disk:
            closure = lambda closure=closure: disk_memoize(self.checkpoints_dir / f"{name}.pt", closure,
                                                           recompute=recompute, log=log, cuda=cuda)
        if in_mem:
            closure = lambda closure=closure: memory_memoize(self.cache, name, closure,
                                                             recompute=recompute, log=log, cuda=cuda)
        return closure()

    # General experiment-related things (directories, files, arguments...).

    @property
    def checkpoints_dir(self) -> Path:
        return global_checkpoints_dir / self.path

    def checkpoint_file(self, epoch=None) -> Path:
        """ Returns the checkpoint file (last epoch by default). """
        suffix = f"_{epoch}" if epoch is not None else ""
        file = f"batchsize_{self.args.batch_size}_lrfreq_{self.args.learning_rate_adjust_frequency}{suffix}.pth.tar"
        return self.checkpoints_dir / file

    @cached_property
    def args(self) -> argparse.Namespace:
        """ Returns args Namespace (new arguments since last experiment are set to their default value). """
        with open(self.checkpoints_dir / "args.json") as f:
            args_dict = json.load(f)
        if "cifar10" in args_dict and args_dict["cifar10"]:  # Backwards compatibility.
            args_dict["dataset"] = "CIFAR10"

        main_block.parser = main_block.build_parser()  # Reset parser because we change the defaults
        main_block.parser.set_defaults(**args_dict)
        args = main_block.get_args([])

        return args

    @property
    def arch(self):
        """ Returns "scat", "resnet", or "vgg". """
        if isinstance(self.args.arch, list):
            return "scat"
        elif self.args.arch.startswith("vgg"):
            return "vgg"
        elif self.args.arch.startswith("resnet"):
            return "resnet"
        else:
            assert False

    # Model-related stuff

    @lru_cache
    def state_dict(self, epoch=None, cuda=True, merge_std=False, log=False):
        """ Returns the saved state_dict with trained weights.
        If merge_std is True, sets the variance parameter of standardization layers to one, and merges it with the next
        learned weight layer. """
        state_dict = models.get_state_dict(checkpoint_file=self.checkpoint_file(epoch=epoch), cuda=cuda, log=log)

        if merge_std:
            eps = 1e-05
            state_dict = state_dict.copy()

            warnings.warn("Warning: merge_std not implemented for classifier")
            for j in self.js[:-1]:  # TODO do this for classifier as well
                var_name = self.var_param_name(j)
                var = state_dict[var_name]

                state_dict[var_name] = torch.ones_like(var) - eps

                p_name = self.p_param_name(j)
                p = state_dict[p_name]
                p = p * torch.rsqrt(var + eps)[None, :, None, None]
                state_dict[p_name] = p

        return state_dict


    def model_with_state(self, state_dict, cuda=True, ignore_errors=False):
        """ Creates a model from a state_dict. """
        return models.get_model(args=self.args, state_dict=state_dict, cuda=cuda, log=False, ignore_errors=ignore_errors)

    @lru_cache
    def model(self, epoch=None, cuda=True, merge_std=False):
        """ Returns the pytorch model with trained weights ready for evaluation. """
        return self.model_with_state(self.state_dict(epoch=epoch, cuda=cuda, merge_std=merge_std), cuda=cuda)

    @lru_cache
    def modules(self, epoch=None, cuda=True, merge_std=False):
        """ Returns a list of Conv2d/ComplexConv2D/Linear modules found in model (including the classifier).
        Convolutions with zero-sized output are ignored. """
        model = self.model(epoch=epoch, cuda=cuda, merge_std=merge_std)
        return models.get_weight_modules(model)

    @property
    def num_layers(self):
        """ Returns the number of learned linear layers. Includes identity conv layers and the classifier. """
        return len(self.modules(epoch=None, cuda=True, merge_std=False))

    @property
    def js(self):
        """ Returns the list of all j in [1, self.num_layers], filtering out identity layers.. """
        if self.arch == "scat":
            # Filter identity matrices.
            sizes = self.args.Pr_size if self.args.Pr_size is not None else self.args.Pc_size
            return [j for j, size in enumerate(sizes, start=1) if size != "id"] + [self.num_layers]
        else:
            return list(range(1, self.num_layers + 1))

    def module(self, j, epoch=None, cuda=True, merge_std=False):
        """ Return a conv/linear module. j is in [1, self.num_layers]. """
        return self.modules(epoch=epoch, cuda=cuda, merge_std=merge_std)[j - 1]

    def mean_param_name(self, j):
        """ Returns the name of the mean parameter of the standardization layer at scale j in state_dict (or None). """
        if self.arch == "scat":
            if j < self.num_layers:
                if 'Std' in self.args.arch:
                    return f"module.{j}.module.{1 + int('rho' in self.args.arch) + self.args.arch.index('Std')}.module.mean"
                else:
                    return None
            else:
                # We need to add one to skip the ToTensor() module.
                return f"module.{self.num_layers + 1}.bn.running_mean"
        elif self.arch in ["vgg", "resnet"]:
            if self.args.standard_batch_norm == "none":
                return None
            elif self.args.standard_batch_norm == "post":
                means = list([k for k in self.state_dict() if "running_mean" in k])
                if 0 <= j - 2 < len(means):
                    return means[j - 2]
                else:
                    return None

    def var_param_name(self, j):
        """ Returns the name of the var parameter of the standardization layer at scale j in state_dict (or None). """
        if self.arch == "scat":
            if j < self.num_layers:
                if 'Std' in self.args.arch:
                    return f"module.{j}.module.{1 + int('rho' in self.args.arch) + self.args.arch.index('Std')}.module.var"
                else:
                    return None
            else:
                # We need to add one to skip the ToTensor() module.
                return f"module.{self.num_layers + 1}.bn.running_var"
        elif self.arch in ["vgg", "resnet"]:
            if self.args.standard_batch_norm == "none":
                return None
            elif self.args.standard_batch_norm == "post":
                vars = list([k for k in self.state_dict() if "running_var" in k])
                if 0 <= j - 2 < len(vars):
                    return vars[j - 2]
                else:
                    return None

    def p_param_name(self, j):
        """ Returns the name of the P_j parameter (N, D) or the classifier (C, D) in state_dict. """
        if self.arch == "scat":
            if j < self.num_layers:
                return f"module.{j}.module.{1 + int('rho' in self.args.arch) + self.args.arch.index('Pr')}.submodules.dict.(0, 0).param"
            else:
                # We need to add one to skip the ToTensor() module.
                return f"module.{self.num_layers + 1}.classifier.weight"
        elif self.arch in ["vgg", "resnet"]:
            weights = list([k for k in self.state_dict() if "weight" in k])
            if j - 1 < len(weights):
                return weights[j - 1]
            else:
                return None

    @property
    def conv_spatial_shapes(self):
        """ Return a dictionary of j -> spatial shape that layer j is acting on.
        Suitable for extracting patches before convolutions. """
        if self.arch != "scat":
            shapes = {}
            for j in self.js:
                module = self.module(j=j)
                if isinstance(module, torch.nn.Conv2d):
                    shape = module.weight.shape[-2:]
                shapes[j] = shape
            return shapes
        else:
            return {j: (1, 1) for j in range(1, self.num_layers)}

    @property
    def linear_spatial_shapes(self):
        """ Return a dictionary of j -> spatial shape that layer j is acting on.
        Suitable for unflattening the input to the classifier. """
        if self.arch != "scat":
            return {}  # There is an avgpool before the classifier.
        else:
            # There is just the spatial shape information before the classifier that has been lost
            # due to flattening at the beginning of the Classifier module.
            return {self.num_layers: models.get_classifier(self.model(epoch=None)).input_type.spatial_shape}

    # Weights-related stuff

    @lru_cache
    def atoms(self, j, epoch=None, merge_std=False, cuda=True):
        """ Returns the weights of P_j as a real or complex (N, D) tensor, or of the classifier as a (C, D) tensor. """
        return models.get_weight_matrix(self.module(j=j, epoch=epoch, merge_std=merge_std, cuda=cuda))

    @lru_cache
    def decomposed_atoms(self, j, epoch=None, merge_std=False):
        """ Returns the weights of P_j as a real or complex (N, D) DecomposedMatrix tensor. """
        return linalg.DecomposedMatrix(self.atoms(j=j, epoch=epoch, merge_std=merge_std), decomposition="svd")

    def atoms_directions(self, j, epoch=None, merge_std=False):
        """ Returns the directions part of Pj, as a (R, D) tensor. """
        return self.decomposed_atoms(j=j, epoch=epoch, merge_std=merge_std).dual_eigenvectors

    @lru_cache
    def atoms_components(self, j, epoch=None, directions_epoch=None, merge_std=False):
        """ Returns the components of atoms in the basis of eigenvectors at the given epoch, as a (R, N) tensor. """
        directions = self.atoms_directions(j=j, epoch=directions_epoch, merge_std=merge_std)
        atoms = self.atoms(j=j, epoch=epoch, merge_std=merge_std)
        return directions @ atoms.T

    @lru_cache
    def atoms_magnitudes(self, j, epoch=None, directions_epoch=None, merge_std=False):
        """ Returns the magnitudes part of Pj, as a (R,) tensor. """
        return torch.linalg.norm(self.atoms_components(
            j=j, epoch=epoch, directions_epoch=directions_epoch, merge_std=merge_std), axis=1)

    @lru_cache
    def atoms_eigenvalues(self, j, epoch=None, merge_std=False):
        """ Returns the eigenvalues of the covariance matrix of the atoms, as a (R,) tensor. """
        return self.decomposed_atoms(j=j, epoch=epoch, merge_std=merge_std).eigenvalues ** 2 / self.num_atoms(j=j)

    @lru_cache
    def atoms_samplings(self, j, epoch=None, directions_epoch=None, merge_std=False):
        """ Returns the sampling part (normalized components) of Pj, as a (R, N) tensor, where R = min(N, D). """
        return self.atoms_components(j=j, epoch=epoch, directions_epoch=directions_epoch, merge_std=merge_std) / \
    self.atoms_magnitudes(j=j, epoch=epoch, directions_epoch=directions_epoch, merge_std=merge_std)[:, None]

    @lru_cache
    def atoms_covariance(self, j, epoch=None, merge_std=False):
        """ Returns the empirical covariance of this network as a (D, D) tensor. """
        return linalg.empirical_covariance(self.atoms(j=j, epoch=epoch, merge_std=merge_std))

    def gaussian_atoms(self, j, epoch=None, num_atoms=None, merge_std=False):
        """ Returns Gaussian atoms (N, D), where N is by default the same as the learned ones. """
        if num_atoms is None:
            num_atoms = self.num_atoms(j=j)
        directions = self.atoms_directions(j=j, epoch=epoch, merge_std=merge_std)  # (R, D)
        magnitudes = self.atoms_magnitudes(j=j, epoch=epoch, merge_std=merge_std) / np.sqrt(self.num_atoms(j=j))  # (R,)
        components = torch.randn((num_atoms, self.num_directions(j=j)), device=directions.device)  # (N, R)
        atoms = components @ (magnitudes[:, None] * directions)  # (N, D)
        return atoms

    def num_atoms(self, j) -> int:
        return self.atoms(j=j, epoch=None).shape[0]

    def dimension(self, j) -> int:
        return self.atoms(j=j, epoch=None).shape[1]

    def num_directions(self, j) -> int:
        return min(self.num_atoms(j), self.dimension(j))

    # Data-related stuff

    def dataloaders(self, batch_size=128, total_classes=1000, sub_classes=None):
        if sub_classes is not None:
            seed = 0
            random_state = np.random.RandomState(seed)
            classes_subset = random_state.choice(total_classes, size=sub_classes, replace=False)
        else:
            classes_subset = None
        return exps.dataloaders(args=self.args, batch_size=batch_size, classes_subset=classes_subset)

    def train_dataloader(self, batch_size=128, total_classes=1000, sub_classes=None):
        return self.dataloaders(batch_size=batch_size, total_classes=total_classes, sub_classes=sub_classes)[0]

    def test_dataloader(self, batch_size=128, total_classes=1000, sub_classes=None):
        return self.dataloaders(batch_size=batch_size, total_classes=total_classes, sub_classes=sub_classes)[1]

    # Activation and alignment stuff

    def activation_filename(self, name, j, other_exp=None, epoch=None, other_epoch=None,
                            standardize=False, dataset_exp=None, test=False) -> str:
        """ Returns filename for activation-related stuff. """
        if isinstance(j, list):
            j = "".join(str(jj) for jj in j)
        filename = f"{name}_j{j}"
        if other_exp is not None:
            filename = f"{filename}_other_{other_exp.name}"
        if epoch is not None:
            filename = f"{filename}_epoch{epoch}"
        if other_epoch is not None:
            filename = f"{filename}_otherepoch{epoch}"
        if standardize:
            filename = f"{filename}_std"
        if dataset_exp is not None:
            filename = f"{filename}_data_{dataset_exp.name}"
        if test:
            filename = f"{filename}_test"

        return filename

    def activation_covariance(self, j, other_exp=None, epoch=None, other_epoch=None, standardize=False,
                              dataset_exp=None, test=False, batch_size=128,
                              recompute=False, log=False, cuda=True, debug=False) -> linalg.DecomposedMatrix:
        """ Returns (D, D') (cross-)covariance matrix as a DecomposedMatrix. Computes on demand, stored on disk.
        j can be a list, in which case this returns a dict of j -> alignment.
        :param standardize: also computes mean and variance of representations across channels
                            (in this case, returns a quintuple (cov, m1_a, m2_a, m1_b, m2_b))
        """

        j_is_list = isinstance(j, list)
        js = j if j_is_list else [j]

        def closure():
            # Other names to avoid nonlocal...
            other = self if other_exp is None else other_exp
            dataset = self if dataset_exp is None else dataset_exp

            models_dict = dict(a=self.model(epoch=epoch, cuda=True, merge_std=False),
                               b=other.model(epoch=other_epoch, cuda=True, merge_std=False))
            pairs = [("a", "b")]
            dataloader = dataset.dataloaders(batch_size=batch_size)[1 if test else 0]
            shapes = self.linear_spatial_shapes
            desc = f"Activation covariance for {self.name + ('' if other_exp is None else f' and {other.name}')} js={js}{'' if dataset_exp is None else f' data={dataset.name}'}{' (test)' if test else ''}"

            res = pairwise_model_covariances(models_dict=models_dict, pairs=pairs, js=js,
                                             dataloader=dataloader, shapes=shapes, compute_std=standardize,
                                             desc=desc, debug=debug)  # "a", "b", j -> cov

            covs = {}
            for j in js:
                # Do ourselves the DecompositionMatrix because pairwise_model_covariances always think it's an SVD
                cov = res["a", "b", j].matrix
                decomposition = "eigh" if self.name == other.name and epoch == other_epoch else "svd"
                cov = linalg.DecomposedMatrix(cov, decomposition=decomposition)

                # Add moments if standardize
                if standardize:
                    cov = cov, res["a", j, 1], res["a", j, 2], res["b", j, 1], res["b", j, 2]

                covs[j] = cov

            return covs if j_is_list else covs[js[0]]

        filename = self.activation_filename(name="act_cov", j=js, other_exp=other_exp, epoch=epoch,
                                            other_epoch=other_epoch, standardize=standardize,
                                            dataset_exp=dataset_exp, test=test)
        return self.memoize(filename, closure, recompute=recompute, log=log, cuda=cuda)

    def activation_covariance_eigenvalues(
        self, j, other_exp=None, epoch=None, other_epoch=None,
        dataset_exp=None, test=False, batch_size=128, recompute=False, log=False) -> torch.Tensor:
        """" Returns (R,) eigenvalues of data (cross-)covariance. """
        def closure():
            cov = self.activation_covariance(j=j, other_exp=other_exp, epoch=epoch, other_epoch=other_epoch,
                                             dataset_exp=dataset_exp, test=test, batch_size=batch_size)
            return cov.eigenvalues

        filename = self.activation_filename(name="act_eigvals", j=j, other_exp=other_exp, epoch=epoch,
                                            other_epoch=other_epoch, dataset_exp=dataset_exp, test=test)
        return self.memoize(filename, closure, recompute=recompute, log=log)

    def alignment(self, j, other_exp=None, epoch=None, other_epoch=None,
                  dataset_exp=None, test=False, batch_size=128, standardize=False,
                  recompute=False, log=False, cuda=True, debug=False) -> torch.Tensor:
        """ Returns (D, D') orthogonal matrix for alignment purposes.
        j can be a list, in which case this returns a dict of j -> alignment.
        :param standardize: if True, computes the alignment between the standardized representations
                            and returns (align, m1, m2, m1ref, m2ref quintuple)
        """

        j_is_list = isinstance(j, list)
        js = j if j_is_list else [j]

        def closure():
            covs = self.activation_covariance(j=js, other_exp=other_exp, epoch=epoch, other_epoch=other_epoch,
                                              dataset_exp=dataset_exp, test=test, batch_size=batch_size,
                                              standardize=standardize, cuda=cuda, debug=debug)
            alignments = {}
            for j, cov in covs.items():
                if standardize:
                    # Take standardization of representations into account.
                    cov, m1, m2, m1ref, m2ref = cov
                    eps = 1e-5  # For numerical stability, matches batch-normalization defaults.
                    c = (cov.matrix - m1[:, None] * m1ref[None, :]) \
                        / torch.sqrt((m2 - m1 ** 2 + eps)[:, None] * (m2ref - m1ref ** 2 + eps)[None, :])
#                     tss = tensor_summary_stats
#                     print(f"{j=} {tss(cov.matrix)=} {tss(m1)=} {tss(m2)=} {tss(m1ref)=} {tss(m2ref)=} {tss(c)=}")
                    cov = linalg.DecomposedMatrix(c, decomposition=cov.decomposition)

                # For numerical stability, enforce identity when we now it should be.
                if other_exp is None \
                    or (j == 1 and (self.args.scat_angles == other_exp.args.scat_angles)) \
                    or (other_exp.name == self.name and epoch == other_epoch):
                    align = torch.eye(cov.matrix.shape[0], device=cov.matrix.device)
                else:
                    align = cov.orthogonalize().matrix

                if standardize:
                    align = align, m1, m2, m1ref, m2ref
                alignments[j] = align

            return alignments if j_is_list else alignments[js[0]]

        filename = self.activation_filename(name=f"alignment_matrix",
                                            j=js, other_exp=other_exp, epoch=epoch, standardize=standardize,
                                            other_epoch=other_epoch, dataset_exp=dataset_exp, test=test)
        return self.memoize(filename, closure, recompute=recompute, log=log, cuda=cuda)

    def alignment_error(self, j, other_exp=None, epoch=None, other_epoch=None, return_full=False, standardize=False,
                        dataset_exp=None, test=True, batch_size=128, recompute=False, log=False, debug=False) -> torch.Tensor:
        """ Returns alignment error as a scalar tensor, measured in space of other experiment.
        j can be "out" for softmax of output of network.
        j can also be a list, in which case this returns a dict of j -> error.
        if standardize, compute alignment error between standardized representations.
        """

        j_is_list = isinstance(j, list)
        js = j if j_is_list else [j]

        def closure():
            # Could use formula from (cross)-covariance singular/eigenvalues but we do it directly (necessary for test set).
            # Other names to avoid nonlocal...
            other = self if other_exp is None else other_exp
            dataset = self if dataset_exp is None else dataset_exp

            models_dict = dict(a=self.model(epoch=epoch, cuda=True, merge_std=False),
                               b=other.model(epoch=other_epoch, cuda=True, merge_std=False))
            modules_dict = {}  # model, j -> module
            align = self.alignment(j=js, other_exp=other_exp, epoch=epoch, other_epoch=other_epoch, standardize=standardize,
                                   dataset_exp=dataset_exp, test=False, batch_size=batch_size, debug=debug)  # j _> (C, C')
            get_align = lambda j: align[j][0] if standardize else align[j]

            for j in js:
                if j != "out":
                    modules_dict["a", j] = self.modules(epoch=epoch, cuda=True, merge_std=False)[j - 1]
                    modules_dict["b", j] = other.modules(epoch=other_epoch, cuda=True, merge_std=False)[j - 1]
                else:
                    modules_dict["a", j] = self.modules(epoch=epoch, cuda=True, merge_std=False)[-1]
                    modules_dict["b", j] = other.modules(epoch=other_epoch, cuda=True, merge_std=False)[-1]

            dataloader = dataset.dataloaders(batch_size=batch_size)[1 if test else 0]

            desc = f"Alignment error for {self.name + ('' if other_exp is None else f' and {other.name}')} js={js}{'' if dataset_exp is None else f' data={dataset.name}'}{' (test)' if test else ''}"

            def to_average(x, y):
                def forward(model):
                    to_save = {j: (("input" if j != "out" else "output"), modules_dict[model, j]) for j in js}
                    activations = models.get_activations(x=x, model=models_dict[model], to_save=to_save)

                    # Reshape and rescale by square-root of dimension, so that total variance is 1 (assumes batch norm).
                    activations = {j: space_to_batch(act, shape=self.linear_spatial_shapes.get(j, None))
                                   for j, act in activations.items()}  # (BMN, C)

#                     for j in js:
#                         print(f"{model=} {j=} act {activations[j].shape} align {tuple(a.shape for a in align[j])}")

                    if standardize:
                        # Standardize activations
                        idx = dict(a=0, b=2)[model]
                        activations = {j: (act - align[j][idx + 1]) / torch.sqrt(align[j][idx + 2] + align[j][idx + 1] ** 2)
                                       for j, act in activations.items()}  # (BMN, C)

                    for act in activations.values():
                        act /= np.sqrt(act.shape[1])
                    return activations

                act_a = forward("a")  # j -> (BMN, C)
                # Realign if necessary
                # TODO: could want to use different arguments here... (regarding dataset, etc)
                act_a = {j: (act @ get_align(j) if j != "out" else act) for j, act in act_a.items()}  # (BMN, C) to (BMN, C')

                act_b = forward("b")  # j -> (BMN, C')

                res = {}
                for j in js:
                    res[j, "aa"] = torch.sum(act_a[j] ** 2, 1)  # (BMN,)
                    res[j, "bb"] = torch.sum(act_b[j] ** 2, 1)  # (BMN,)
                    res[j, "ab"] = torch.sum(act_a[j] * act_b[j], 1)  # (BMN,)
#                     res[j] = res[j, "aa"] + res[j, "bb"] - 2 * res[j, "ab"]

                return res

            errors = expected_value(to_average, dataloader, desc=desc)  # j -> error
            return errors if j_is_list else errors[js[0]]

        filename = self.activation_filename(name="alignment_error", j=js, other_exp=other_exp, epoch=epoch,
                                            other_epoch=other_epoch, standardize=standardize, dataset_exp=dataset_exp,
                                            test=test)
        return self.memoize(filename, closure, recompute=recompute, log=log)

    # Old stuff

    @lru_cache
    def reshaped_weight_matrices(self, epoch=None):
        """ Weights reshaped to be (N, N_prev, 1+L). """
        Ps = []
        N_prev = 1
        for P in self.weight_matrices(epoch=epoch):
            Ps.append(models.reshape_atoms(P, N_prev))
            N_prev = N

        return Ps

    @cached_property
    def visualizations(self):
        """ Return activation images for each atom, as a (K, 32, 32) array. """
        from .lucent_hack import visualize_atom
        return visualize_atom(model=self.model, layer=f"module_1_module_3_submodules_dict_(0, 0)",
                              atom=np.arange(self.num_atoms), batch=1, stationary=True, show=False)



class Experiments:
    """ Singleton class which holds several experiments.
    Experiments are stored in a dictionary, but also as attributes for easy auto-completion.
    """
    def __init__(self):
        self.exps: Dict[str, Experiment] = {}  # exp_name -> Experiment
        self.all_exps_cache = {}  # subdir -> sorted list of all experiments in dir
        self.imagenet_path = "/mnt/home/fguth/sparse_scat_net/data/2012"
        self.logfile = None  # Optional logfile for collecting logs when run outside a notebook (should be an open file).

    def all_exps(self, subdir="", reload=False):
        """ Returns a sorted list of all experiments on disk. """
        if reload or subdir not in self.all_exps_cache:
            self.all_exps_cache[subdir] = sorted(list((global_checkpoints_dir / subdir).iterdir()))
        return self.all_exps_cache[subdir]

    def load(self, path: Union[str, Path], reload=False, log=True) -> Experiment:
        """ Loads (or reload) an experiment into self. path is the suffix to append to global_checkpoints_dir """
        name = path_to_name(path)
        if reload or name not in self.exps:
            if log:
                print_and_write(f"Loading experiment {path}", self.logfile)
            exp = Experiment(exps=self, path=path)
            self.exps[name] = exp
            setattr(self, exp.name.replace('-', '_'), exp)
        else:
            if log:
                print_and_write(f"Experiment {path} is already loaded", self.logfile)
        return self.exps[name]

    def load_pattern(self, pattern: str, subdir="", reraise=True, reload=False,
                     full_names=True, log=True, single=False, num=None) -> Union[Dict[str, Experiment], Experiment]:
        """ Loads several experiments into self. """
        regexp = re.compile(pattern)
        subdir = Path(subdir)
        exps = {}
        for exp_name in self.all_exps(subdir=subdir, reload=reload):
            if regexp.search(str(exp_name.name)):
                try:
                    exp = self.load(subdir / exp_name.name, reload=reload, log=log)
                    exps[exp.name] = exp
                    if single or len(exps) == num:
                        break
                except Exception as e:
                    print_and_write(f"Ignored {exp_name.name}: {e.__class__.__name__} {e}", self.logfile)
                    if reraise:
                        raise e
        print_and_write(f"Found {len(exps)} exps matching {pattern}", self.logfile)

        if single:
            return list(exps.values())[0]
        if not full_names:
            long2short = short_names(exps.keys())
            exps = {long2short[name]: exp for name, exp in exps.items()}
        return exps

    def remove(self, exp_name: str):
        """ Remove an experiment from self. """
        print_and_write(f"Removing experiment {exp_name}", self.logfile)
        del self.exps[exp_name]
        delattr(self, exp_name.replace('-', '_'))

    def remove_pattern(self, pattern: str):
        """ Removes several experiments from self. """
        regexp = re.compile(pattern)
        for exp_name in list(self.exps.keys()):
            if regexp.search(exp_name):
                self.remove(exp_name)

    def clear(self):
        """ Removes all experiments from self. """
        for exp_name in self.exps.copy():
            self.remove(exp_name)

    def subset(self, pattern: str) -> Dict[str, Experiment]:
        """ Filter a subset of experiments whose name contain the given pattern (regular expression). """
        regexp = re.compile(pattern)
        res = {exp_name: exp for exp_name, exp in self.exps.items() if regexp.search(exp_name)}
        print_and_write(f"Found {len(res)} exps matching {pattern}", self.logfile)
        return res

    def arg_to_regexp(self, arg, default=r"\d+"):
        if arg is None:
            return default
        elif isinstance(arg, list):
            return f"({'|'.join(str(a) for a in arg)})"
        else:
            return str(arg)

    def get_P_regexp(self, j=1, L=32, num_atoms=None, suffix=""):
        """ num_atoms can be a list. None means any learned P, 0 means identity P. """
        def atom_to_regexp(atom):
            if atom == 0:
                return ''
            else:
                return f"P{self.arg_to_regexp(atom)}"

        if not isinstance(num_atoms, list):
            num_atoms = [num_atoms] * j
        num_atoms = ''.join(['W' + atom_to_regexp(atom) for atom in num_atoms])

        angles = ''.join('L' + self.arg_to_regexp(l) for l in L)

        regexp = f"-{num_atoms}W-{angles}-{suffix}"
        print_and_write(f"Searching with regexp {regexp}", self.logfile)
        return regexp

    def load_P_exps(self, j=1, L=32, num_atoms=None, suffix="", reload=False):
        """ Wrapper for load_pattern with regexp generator. """
        self.load_pattern(self.get_P_regexp(j=j, L=L, num_atoms=num_atoms, suffix=suffix), reload=reload)

    def P_subset(self, j=1, L=32, num_atoms=None, suffix=""):
        """ Wrapper for subset with regexp generator. """
        return self.subset(self.get_P_regexp(j=j, L=L, num_atoms=num_atoms, suffix=suffix))

    @lru_cache
    def _dataloaders(self, dataset, grayscale, image_size, randomize_labels, classes_subset, data_subset, batch_size, num_workers):
        """ Returns the training and validation dataloaders. """
        train_dataset, val_dataset = datasets.get_datasets(
            dataset_name=dataset, grayscale=grayscale, imagenet_path=self.imagenet_path,
            image_size=image_size, randomize_label_seed=randomize_labels,
        )

        train_dataset, val_dataset = datasets.take_subset(
            train_dataset=train_dataset, val_dataset=val_dataset,
            classes_subset=classes_subset, data_subset=data_subset,
        )

        def get_dataloader(dataset, shuffle):
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, pin_memory=True)

        train_loader = get_dataloader(train_dataset, shuffle=True)
        val_loader = get_dataloader(val_dataset, shuffle=False)
        return train_loader, val_loader

    def dataloaders(self, args, batch_size=128, classes_subset=None):
        """ batch_size, classes_subset override defaults in arguments. """
        if classes_subset is None:
            classes_subset = args.classes_subset
        if classes_subset is not None:
            classes_subset = tuple(classes_subset)
        return self._dataloaders(
            dataset=args.dataset, grayscale=args.grayscale, image_size=args.resize_images,
            randomize_labels=args.randomize_labels, classes_subset=classes_subset, data_subset=args.data_subset,
            batch_size=batch_size, num_workers=2,
        )

    @cached_property
    def class_labels(self):
        """ Labels of the classes of the default dataset (CIFAR10). """
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    @property
    def short_names(self) -> Dict[str, str]:
        """ Return short names computed from all loaded experiment names. """
        return short_names(self.keys())

    def __iter__(self):
        return self.exps.__iter__()

    def __len__(self):
        return self.exps.__len__()

    def keys(self):
        return self.exps.keys()

    def values(self):
        return self.exps.values()

    def items(self):
        return self.exps.items()

    def __getitem__(self, item):
        return self.exps.__getitem__(item)


def short_names(names: Iterable[str]) -> Dict[str, str]:
    """ Compute short names from a list of experiments, by removing common terms in experiment names.
    Returns a dictionary of long names to short names. """
    terms_list = {name: name.split('-') for name in names}
    terms_set = {name: set(terms) for name, terms in terms_list.items()}
    all_terms = set().union(*terms_set.values())
    common_terms = {term for term in all_terms if all(term in terms for terms in terms_set.values())}
    unique_terms = {name: [term for term in terms if term not in common_terms]
                    for name, terms in terms_list.items()}
    return {name: '-'.join(terms) for name, terms in unique_terms.items()}


def memoize(closure: Callable[[], Any], path: Union[str, Path], recompute=False, log=False) -> Any:
    """ Memoizes the result of closure to disk with the given path. """
    if not isinstance(path, Path):
        path = Path(path)
    if path.exists() and not recompute:
        result = torch.load(str(path))
        if log:
            print_and_write(f"Loaded {path}", exps.logfile)
    else:
        result = closure()
        torch.save(result, str(path))
        if log:
            print_and_write(f"Computed and saved {path}", exps.logfile)
    return result


def space_to_batch(x, shape=None, kernel_size=None, stride=1, padding=0, dilation=1):
    """
    :param x: (B, C') or (B, C, M, N)
    :param shape: optional (M, N) spatial shape to give to 2d input
    :param kernel_size: optional (K, K) patch size to extract from 2d input
    :return: (BP, CK²) where P is the number of patches extracted from (M, N) (defined by input shape or shape param)
    """
    # Do a reshape of spatial shape.
    if shape is None:
        if x.ndim == 2:
            shape = (1, 1)
        elif x.ndim == 4:
            shape = x.shape[-2:]  # (M, N)
        else:
            assert False

    # Permute if there is a spatial shape.
    # Alternative is shape was given as (1, 1) or None and x is 2D
    if shape != (1, 1):
        x = x.reshape((x.shape[0], -1) + shape)  # (B, C, M, N)
        # Now extract patches
        if kernel_size not in [None, 1, (1, 1)]:
            x = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=stride,
                                           padding=padding, dilation=dilation)  # (B, CK², P)
            x = x.permute(0, 2, 1)  # (B, P, CK^2)
        else:
            x = x.permute(0, 2, 3, 1)  # (B, M, N, C)

    # Flatten and return.
    return x.reshape((-1, x.shape[-1]))  # (BMN, C) or (BP, CK^2)


def flatten_space(x):
    """ (B, C, M, N) to (B, CMN) """
    return x.reshape((x.shape[0], -1))


def expected_value(closure: Callable[[torch.Tensor], Dict[Any, Tuple[torch.Tensor, int]]],
                   dataloader, desc=None,
                  ) -> Dict[Any, torch.Tensor]:
    """ Compute an expected value of several functions of input data over the whole dataset.
    functions should return either one (B, *) or two (B, C), (B, C') tensors, and we compute
    either the mean (*,) or the second moment (C, C').
    This allows for computing covariances in a more memory-efficient manner.
    :param closure: closure(x, y) where x is of shape (B, C, M, N) returns a dict: key -> one or two Tensors
    :param dataloader: dataloader to use
    :param desc: optional description for tqdm
    :return: key -> (*_key)
    TODO: option to compute class means?
    """
    sums = {}  # key -> (sum of tensors, sum of counts).

    for x, y in tqdm(dataloader, desc=desc):
        for key, res in closure(x.cuda(), y.cuda()).items():
            # Ugly implementation to optimize memory (in-place additions).
            if isinstance(res, tuple):
                # Compute second moment.
                a, b = res  # (B, C), (B, C')
                count = a.shape[0]
                if key in sums:
                    prev_tensor, prev_count = sums[key]
                    prev_tensor.addmm_(a.H, b)
                    sums[key] = prev_tensor, prev_count + count
                else:
                    sums[key] = a.T @ b, count
            else:
                # Compute first moment.
                a = res  # (B, *)
                count = a.shape[0]
                if key in sums:
                    prev_tensor, prev_count = sums[key]
                    prev_tensor += a.sum(0)
                    sums[key] = prev_tensor, prev_count + count
                else:
                    sums[key] = a.sum(0), count

    # Also ugly code for in-place division.
    means = {}
    for key, (tensor, count) in sums.items():
        tensor /= count
        means[key] = tensor
    return means


def pairwise_model_covariances(models_dict: Dict[Any, torch.nn.Module], pairs: List[Tuple[Any, Any]],
                               js, dataloader, shapes=None, patches=False, compute_std=False, debug=False, real=True,
                               additional: Callable[[Dict[Any, Dict[int, torch.Tensor]]],
                                                    Dict[Any, Dict[int, torch.Tensor]]]=None,
                               desc=None,
                              ) -> Dict[Tuple[Any, Any, int], linalg.DecomposedMatrix]:
    """ Computes pairwise covariances between models.
    Chosen pairs should have the same spatial resolution at each layer for the channel-wise covariances.
    Complex activations are dealt with, but only the real part of covariances is returned for simplicity.
    :param models_dict: dictionary of models to compute activations of
    :param pairs: lists of pairs for the cross-covariance computations
    :param js: list of layers to compute covariances at
    :param dataloader: dataloader to use
    :param shapes: optional dict from j to new spatial shape (M, N) or None (for no action)
    :param patches: whether to compute covariance over patches or not
    :param compute_std: additionally estimate per-channel mean and variance, as (model, j, m) -> moment of order m
    :param debug: whether to print information regarding shapes of tensors for debugging OOM errors
    :param real: if True, takes the real part of complex covariances
    :param additional: function which adds additional "aggregate" outputs.
    :param desc: optional description for tqdm
    Input is dictionary key -> j -> output of model[key] on layer j, output is additional key -> j -> output dict.
    Additional keys need to appear in pairs to be taken into account.
    :return: dictionary (model1, model2, j) -> cross-covariance between inputs to Pj for model1 and model2
    """
    if shapes is None:
        shapes = {}
    if patches is None:
        patches = {}

    def closure(x, y):
        outputs = {}  # key -> j -> (BM[j]N[j], C[key,j])
        for key, model in models_dict.items():
            modules = models.get_weight_modules(model)
            to_save = {j: ("input", modules[j - 1]) for j in js}
            activations = models.get_activations(x=x, model=model, to_save=to_save)

            outputs_key = {}
            for j, z in activations.items():
                shape = shapes.get(j, None)

                module = modules[j - 1]
                kwargs = {}
                if patches and isinstance(module, torch.nn.Conv2d):
                    kwargs.update(kernel_size=module.kernel_size, stride=module.stride,
                                  padding=module.stride, dilation=module.dilation)

                outputs_key[j] = space_to_batch(z, shape=shape, **kwargs)  # (BMN, C)
                if debug:
                    print(f"{key=} {j=} output {z.shape} spatial {shape} kwargs {kwargs} to {outputs_key[j].shape}")
            outputs[key] = outputs_key

        if additional is not None:
            outputs.update(additional(outputs))

        moments = {}

        # Covariances
        for key1, key2 in pairs:
            for j in js:
                if debug:
                    s1 = tuple(outputs[key1][j].shape)
                    s2 = tuple(outputs[key2][j].shape)
                    s3 = (s1[1], s2[1])
                    print_and_write(f"Multiplying {j=} {key1=} {s1=} with {key2=} {s2=} to give {s3=}", exps.logfile)

                moments[key1, key2, j] = outputs[key1][j], outputs[key2][j]  # will be (C1, C2)

        # Standardization
        if compute_std:
            for key, output in outputs.items():
                for j in js:
                    moments[key, j, 1] = output[j]  # will be (C,)
                    moments[key, j, 2] = torch.abs(output[j]) ** 2  # will be (C,)

        return moments

    if desc is None:
        desc = f"Activation covariances: {len(models_dict)} models, {len(pairs)} pairs and js = {js}"
    covs = expected_value(closure, dataloader, desc=desc)

    # Convert to DecomposedMatrix, and add symmetric part for convenience.
    for key1, key2, in pairs:
        for j in js:
            c = covs[key1, key2, j]
            if real:
                c = torch.real(c)
            c = linalg.DecomposedMatrix(c, decomposition="eigh" if key1 == key2 else "svd")  # (C1, C2)
            covs[key1, key2, j] = c
            if key1 != key2:
                covs[key2, key1, j] = c.T  # (C2, C1)

    return covs


def cross_experiment_covariances(experiments: Dict[Any, Experiment], dataloader=None, batch_size=1024, dataset_key=None,
                                 cross_experiments=None, epochs=None, cross_epochs=None, js=None, shapes=None, recompute=False,
                                ):
    """ Computes cross-covariances between models from different experiments.
    :param experiments: experiments to use models from
    :param dataset_key: key to identify the dataset used to compute covariances
    :param dataloader: dataloader to use (by default: train dataloader of first experiment)
    :param cross_experiments: can be: - None (no cross-covariances)
                                      - "first" (cross-covariances with first experiment)
                                      - "full" (cross-covariances between all experiments)
    :param epochs: optional list of epochs to consider
    :param cross_epochs: can be: - None (no cross-covariances)
                                 - "first": (cross-covariances with initialization)
                                 - "last": (cross-covariances with trained network)
                                 - "full": (cross-covariances between all epochs)
    :param js: optional list of layers to align
    :param shapes: optional dict from j to new spatial shape (M, N) or None (for no action)
    """
    if dataloader is None:
        dataloader = list(experiments.values())[0].train_dataloader(batch_size=batch_size)

    if epochs is None:
        epochs = [None]

    exp_keys = list(experiments.keys())
    if cross_experiments is None:
        exp_key_pairs = [(key, key) for key in exp_keys]
    elif cross_experiments == "first":
        exp_key_pairs = [(exp_keys[0], key) for key in exp_keys]
    elif cross_experiments == "full":
        exp_key_pairs = [(exp_keys[i], exp_keys[j]) for i in range(len(exp_keys)) for j in range(i, len(exp_keys))]

    if cross_epochs is None:
        epoch_pairs = [(epoch, epoch) for epoch in epochs]
    elif cross_epochs == "first":
        epoch_pairs = [(0, epoch) for epoch in epochs]
    elif cross_epochs == "last":
        epoch_pairs = [(None, epoch) for epoch in epochs]
    elif cross_epochs == "full":
        epoch_pairs = [(epochs[i], epochs[j]) for i in range(len(epochs)) for j in range(i, len(epochs))]

    if js is None:
        js = experiments[exp_keys[0]].js
    if shapes is None:
        shapes = experiments[exp_keys[0]].linear_spatial_shapes

    pairs = []
    for key1, key2 in exp_key_pairs:
        for epoch1, epoch2 in epoch_pairs:
            # Filter pairs that we have already computed.
            if recompute or any((dataset_key, epoch1, key2, epoch2, j) not in experiments[key1].data_covariances for j in js):
                pairs.append(((key1, epoch1), (key2, epoch2)))

    if len(pairs) == 0:
        return

    models = {(key, epoch): exp.model(epoch=epoch, cuda=True, merge_std=False) for key, exp in experiments.items() for epoch in epochs}
    covs = pairwise_model_covariances(models_dict=models, pairs=pairs, js=js, dataloader=dataloader, shapes=shapes)
    for ((key1, epoch1), (key2, epoch2), j), cov in covs.items():
        experiments[key1].data_covariances[dataset_key, epoch1, key2, epoch2, j] = cov


class AlignedExperiments:
    """ Class to hold several experiments from different initializations.
    Stores the alignment matrices and handles the covariance logic.
    """
    def __init__(self, exps: Dict[Any, Experiment], ref_exp=None, name: str = "",
                 num_iters=2, js=None, merge_std=False,
                 batch_size=128, recompute=False, log=False):
        """
        :param exps: experiments to align together
        :param ref_exp: optional experiment to align to
        :param name: name of the experiments set for caching purposes.
        In most cases this is not needed, as we append the number of experiments, num_iters, etc.
        Files are stored in the dir of the first exp.
        :param num_iters: number of iterations for the alignment: -1 to skip
        :param js: optional list of layers to align (default: all, including classifier)
        :param merge_std: whether to use merged models for alignment
        """
        self.num_exps = len(exps)
        self.exps_dict = exps  # key -> Experiment
        self.keys = list(exps.keys())
        self.exps_list = list(exps.values())
        # Default exp and key.
        self.key = self.keys[0]
        self.exp = self.exps_list[0]

        self.ref_exp = ref_exp
        self.ref_name = None if self.ref_exp is None else self.ref_exp.name

        self.num_iters = num_iters
        self.js = self.exp.js if js is None else js  # list of layers to align
        self.merge_std = merge_std
        self.conv_spatial_shapes = self.exp.conv_spatial_shapes  # j -> spatial shape to unflatten weights of convolution
        self.linear_spatial_shapes = self.exp.linear_spatial_shapes  # j -> spatial shape to unflatten input of classifier
        self.all_spatial_shapes = {**self.conv_spatial_shapes, **self.linear_spatial_shapes}

        if self.ref_exp is None:
            self.name = f"{name}-exps{self.num_exps}-iters{self.num_iters}-js{''.join(str(j) for j in self.js)}"
        else:
            # NOTE: this does not contain self.key, which could cause cache collisions in some (unprobable) cases.
            self.name = f"{self.ref_name}_{name}_exps{self.num_exps}"
        if merge_std:
            self.name = f"{self.name}-mergestd"

        self.mean_key = "mean"  # f"{self.name}-mean"

        # Multi-alignment
        if self.ref_exp is None and self.num_iters >= 0:
            self.dataloader = self.exp.train_dataloader(batch_size=batch_size)
            self.test_dataloader = self.exp.test_dataloader(batch_size=batch_size)

        self.mean_alignments = None  # key, j -> (D_key, D_mean) orthogonal matrix
        self.total_variance = None  # j -> () total variance
        self.unexplained_variance = []  # iter -> j -> fraction of total variance at given iteration
        self.data_covariances = None  # key, j -> (D_key, D_key) DecomposedMatrix
        self.cache = {}

        if self.ref_exp is None:
            self.compute_alignment(recompute=recompute, log=log)

    def clear(self):
        """ Empties cache to free memory. Also empties cache of associated experiments. """
        self.cache.clear()
        for exp in self.exps_list:
            exp.clear()

    def memoize(self, name, closure, in_mem=True, on_disk=True, recompute=False, log=False, cuda=True):
        """ Memoizes a computation in memory (self.cache) and/or on disk (checkpoint_dir of first exp). """
        name = f"{self.name}_{name}"
        if on_disk:
            closure = lambda closure=closure: disk_memoize(
                self.exp.checkpoints_dir / f"{name}.pt", closure, recompute=recompute, log=log, cuda=cuda)
        if in_mem:
            closure = lambda closure=closure: memory_memoize(
                self.cache, name, closure, recompute=recompute, log=log, cuda=cuda)
        return closure()

    def compute_alignment(self, recompute=False, log=True):
        """ Compute alignment matrices to the mean representation, along with the mean representation. """
        def closure():
            models = self.models()
            """ Algorithm:
            - Step 0: align everything to the first network and define the first real mean network, compute total variance
            - Further steps: Re-align to this mean network, compute explained variance of the previous step
            """

            for i in range(self.num_iters + 1):
                # Compute covariances of models with mean.
                pairs = [(key, self.mean_key) for key in self.keys]
                if i == 0:
                    # Compute total variance for upper bound.
                    pairs.extend([(key, key) for key in self.keys])
                # Compute mean variance for explained variance.
                pairs.append((self.mean_key, self.mean_key))
                covs = pairwise_model_covariances(
                    models_dict=models, pairs=pairs, js=self.js, dataloader=self.dataloader,
                    shapes=self.linear_spatial_shapes, additional=self.mean_network(),
                )  # key, "mean", j -> cov of shape (D_key, D_mean)

                # Compute explained variance of the previous alignments.
                if i == 0:
                    self.data_covariances = {(key, j): covs[key, key, j] for key in self.keys + [self.mean_key] for j in self.js}
                    self.total_variance = {j: sum(torch.trace(covs[key, key, j].matrix) for key in self.keys).item() / self.num_exps
                                           for j in self.js}
                else:
                    self.data_covariances.update({(self.mean_key, j): covs[self.mean_key, self.mean_key, j] for j in self.js})
                    correlation = {j: torch.trace(covs[self.mean_key, self.mean_key, j].matrix).item() for j in self.js}
                    self.unexplained_variance.append({j: 1 - correlation[j] / self.total_variance[j]
                                                      for j in self.js})
                    self.report_alignment_performance()

                # Compute new alignment matrices of models with mean.
                self.mean_alignments = {(key, j): covs[key, self.mean_key, j].orthogonalize().matrix
                                        for key in self.keys for j in self.js}

            return self.mean_alignments, self.total_variance, self.unexplained_variance, self.data_covariances  # Needed for proper memoizing.

        if self.num_exps == 1 and self.num_iters > 0:
            # Do one iteration of "alignment" to compute data covariances
            self.num_iters = 0

        if self.num_iters >= 0:
            # Slightly inefficient for n = 2, because only one step is needed...
            self.mean_alignments, self.total_variance, self.unexplained_variance, self.data_covariances = memoize(
                closure, path=self.exp.checkpoints_dir / f"alignments-{self.name}",
                recompute=recompute, log=log,
            )
        else:
            self.mean_alignments = {(self.keys[0], j):  torch.eye(
                self.exp.dimension(j=j) // (np.prod(self.all_spatial_shapes[j]) if j in self.all_spatial_shapes else 1),
                dtype=self.exp.atoms(j=j).dtype, device=self.exp.atoms(j=j).device) for j in self.js}

    def models(self, epoch=None, cuda=True) -> Dict[Any, torch.nn.Module]:
        return {key: exp.model(epoch=epoch, cuda=cuda, merge_std=self.merge_std) for key, exp in self.exps_dict.items()}

    def mean_network(self, prefix="", mean_key=None):
        """ Returns a closure suitable for expected value computations.
        :param prefix: prefix in front of all keys, both individual exps and mean
        :param mean_key: optional key for the mean, defaults to mean_key
        :return: dictionary with the mean output, key is concatenation of prefix and mean_key
        """
        k = lambda key: f"{prefix}{key}"
        if mean_key is None:
            mean_key = self.mean_key

        if self.mean_alignments is None or self.num_exps == 1:
            # Initial aggregate or single experiment: return the output of the first network
            def additional(outputs):
                return {k(mean_key): outputs[k(self.key)]}
        else:
            def additional(outputs):
                js = list(outputs[k(self.key)].keys())
                aligned_outputs = {key: {j: outputs[k(key)][j] @ self.mean_alignments[key, j] for j in js}
                                   for key in self.keys}  # (B, C_key) -> (B, C_mean)
                mean_output = {j: sum(aligned_outputs[key][j] for key in self.keys) / self.num_exps
                               for j in js}
                return {k(mean_key): mean_output}
        return additional

    def report_alignment_performance(self):
        """ Prints a breakdown of alignment performance. """
        def explained_dimensions(j):
            """ Computes the number of dimension explained by a given variance.
            This is done by comparing the explained variance (variance of the mean representation) with the number of
            principal directions needed to reach that variance in any single network
            (which is arbitrary because the spectrum concentrates).
            """
            explained_variance = torch.trace(self.data_covariances[self.mean_key, j].matrix).item()
            cumulative_variance = torch.cumsum(self.data_covariances[self.key, j].eigenvalues, dim=0)  # (R,)
            dimensions = (cumulative_variance < explained_variance).sum()  # ()
            return f"{dimensions} explained dims out of {cumulative_variance.shape[0]}"

        print_and_write(f"Step {len(self.unexplained_variance)}: unexplained variance {' '.join(f'{j=} {self.unexplained_variance[-1][j]:.3%} ({explained_dimensions(j)})' for j in self.js)}", exps.logfile)

    def align_weight(self, weight, alignment, j):
        """ Align weight matrix (C, D) or (C, DMN) with alignment matrix (D, D') to give (C, D'MN). """
        if j in self.all_spatial_shapes:
            weight = weight.reshape((weight.shape[0], -1) + self.all_spatial_shapes[j])  # (C, D, M, N)
            weight = linalg.contract(weight, alignment.T, axis=1)  # (C, D', M, N)
            weight = weight.reshape((weight.shape[0], -1))  # (C, D'MN)
        else:
            weight = weight @ alignment  # (C, D')
        return weight

    def alignment(self, j, key, epoch=None, log=False, cuda=True):
        """ Returns (D_key, D_ref) alignment matrix to mean/reference network.
        epoch is the epoch at which alignment is computed (both networks are taken at this epoch).
        """
        if self.mean_alignments is not None:
            # Old API, alignment to mean network. Can't deal with several epochs, so requires last epoch or single experiment.
            assert epoch is None or self.num_exps == 1
            return self.mean_alignments[key, j]
        else:
            return self.exps_dict[key].alignment(j=j, other_exp=self.ref_exp, epoch=epoch, other_epoch=epoch,
                                                 log=log, cuda=cuda)

    def aligned_weight(self, j, key, epoch=None, align_epoch=None,
                       cache=True, log=False, cuda=True):
        """ Returns the aligned weight of a given exp in mean space, as a (N, D_mean) tensor.
        :param epoch: epoch of the weights
        :param align_epoch: epoch at which alignment is computed (for both networks)
        """
        closure = lambda: self.align_weight(
            self.exps_dict[key].atoms(j=j, epoch=epoch, merge_std=self.merge_std, cuda=cuda),
            self.alignment(j, key, epoch=align_epoch, log=log, cuda=cuda), j)
        return self.memoize(f"aligned_weight_j{j}_{key}_epoch{epoch}_alignepoch{align_epoch}",
                            closure, on_disk=False, in_mem=cache, log=log, cuda=cuda)

    def aligned_weights(self, j, cat=True, epoch=None, align_epoch=None,
                        cache=True, log=False, cuda=True):
        """ Returns all the aligned weights in mean space, as (num_exps, num_atoms, D) or (N_tot, D) tensor.
        :param cat: whether to concatenate or stack the atoms (stack only supported for same-width experiments)
        :param epoch: epoch of the weights
        :param align_epoch: epoch at which alignment is computed (for both networks)
        :return: atoms from all experiments in mean space, shape (num_exps, num_atoms, D) (cat=False) or (N_tot, D) (cat=True).
        """
        method = torch.cat if cat else torch.stack
        closure = lambda: method([self.aligned_weight(j=j, key=key, epoch=epoch, align_epoch=align_epoch,
                                                      cache=False, log=log, cuda=cuda)
                                 for key in self.keys], dim=0)  # (num_exps, num_atoms, D_mean) or (N_tot, D_mean)
        return self.memoize(f"aligned_weights_j{j}_cat{cat}_epoch{epoch}_alignepoch{align_epoch}",
                            closure, on_disk=False, in_mem=cache, log=log, cuda=cuda)

    def covariance(self, j, epoch=None, align_epoch=None, num_atoms=None, rank=None, key=None,
                   log=False, cuda=True) -> linalg.DecomposedMatrix:
        """ Returns the covariance in mean space, (D_mean, D_mean).
        Optionally restrict the number of atoms, the rank, or use a single experiment.
        :param epoch: epoch of the weights
        :param align_epoch: epoch at which alignment is computed (for both networks)
        :param num_atoms: optionally restrict the number of atoms used to compute the covariance
        :param rank: optionally restrict the rank of the covariance (requires diagonalization)
        :param key: optionally compute the covariance of a single experiment (still aligned)
        """
        def closure():
            if key is None:
                atoms = self.aligned_weights(j=j, cat=True, epoch=epoch, align_epoch=align_epoch,
                                             cache=False, log=log, cuda=cuda)  # (N_tot, D)
            else:
                atoms = self.aligned_weight(j=j, key=key, epoch=epoch, align_epoch=align_epoch,
                                            cache=False, log=log, cuda=cuda)  # (N, D)
            if num_atoms is not None:
                atoms = atoms[:num_atoms]
            return linalg.empirical_covariance(atoms)

        name = f"weight_cov_j{j}_epoch{epoch}_alignepoch{align_epoch}_num{num_atoms}"
        if key is not None:
            name = f"{name}_key_{short_hash(key)}"
        return self.memoize(name, closure, on_disk=True, in_mem=True, log=log, cuda=cuda).project(rank=rank)

    def covariance_error(self, j, ref_aligned, epoch=None, align_epoch=None, num_atoms=None, rank=None,
                         p="inf", renorm=False, key_self=None, key_ref=None,
                         recompute=False, log=False, cuda=True) -> torch.Tensor:
        """ Returns the error between the estimated covariance and the one of the ref_aligned.
        Note that self and ref_aligned should have the same reference experiment.
        NOTE: this could be optimized when rank is small, by never representing big matrices.
        :param j:
        :param ref_aligned: AlignedExperiments, to compare the covariance to
        :param epoch, align_epoch, num_atoms, rank: given to both covariances
        :param p: which p-Schatten norm to use
        :param renorm: whether to divide the covariance by their norms
        :param key_self, key_ref: optionally compute only the error between single covariances rather than averaged ones
        """
        assert self.ref_name == ref_aligned.ref_name
        def closure():
            ord = {1: "nuc", 2: "fro", "inf": 2}[p]  # Translate p to ord parameter of matrix norm
            def cov(aligned, key):
                c = aligned.covariance(j=j, epoch=epoch, align_epoch=align_epoch, num_atoms=num_atoms, rank=rank, key=key,
                                       log=log, cuda=cuda)
                if renorm:
                    factor = 1 / torch.linalg.matrix_norm(c.matrix)
                else:
                    factor = aligned.exp.dimension(j=j)
                return c.matrix * factor
            c1 = cov(self, key_self)
            c2 = cov(ref_aligned, key_ref)
            return torch.linalg.matrix_norm(c1 - c2, ord=ord)

        name = f"weight_cov_error_j{j}_epoch{epoch}_alignepoch{align_epoch}_num{num_atoms}"
        if rank is not None:
            name = f"{name}_rank{rank}"
        if p != "inf":
            name = f"{name}_p{p}"
        if renorm:
            name = f"{name}_renorm"
        if key_self is not None:
            name = f"{name}_keyself_{short_hash(key_self)}"
        if key_ref is not None:
            name = f"{name}_keyref_{short_hash(key_ref)}"
        # ref_aligned.name does not provide redundant information, as it might be different from self.ref_name
        # Only hash ref_aligned.name if a key is provided (backwards compatibility...).
        name = f"{name}_ref_{ref_aligned.name if key_self is None else short_hash(ref_aligned.name)}"
        return self.memoize(name, closure, on_disk=True, in_mem=True, recompute=recompute, log=log, cuda=cuda)

    def covariance_eigenvalues(self, j, epoch=None, align_epoch=None, num_atoms=None,
                               log=False, cuda=True) -> linalg.DecomposedMatrix:
        """ Returns the spectrum of the covariance (R,). Optionnaly restrict the number of atoms.
        :param epoch: epoch of the weights
        :param align_epoch: epoch at which alignment is computed (for both networks)
        """
        closure = lambda: self.covariance(j=j, epoch=epoch, align_epoch=align_epoch, num_atoms=num_atoms,
                                          log=log, cuda=cuda).eigenvalues
        return self.memoize(f"weight_eigenvals_j{j}_epoch{epoch}_alignepoch{align_epoch}_num{num_atoms}",
                            closure, on_disk=True, in_mem=True, log=log, cuda=cuda)

    @lru_cache
    def pca_weight(self, j, key, w_epoch=None, c_epoch=None):
        """ Returns the weight of a given exp in PCA space, as a (N, R) tensor.
        :param w_epoch: the epoch of the weights
        :param c_epoch: the epoch of the covariance
        """
        return self.aligned_weight(j=j, key=key, epoch=w_epoch) @ self.covariance(j=j, epoch=c_epoch).eigenvectors.T   # (N, R)

    @lru_cache
    def pca_weight_norm(self, j, key, w_epoch=None, c_epoch=None):
        """ Returns the norm of the weight of a given exp in PCA space, as a (R,) tensor. """
        return torch.linalg.norm(self.pca_weight(j=j, key=key, w_epoch=w_epoch, c_epoch=c_epoch), dim=0)  # (R,)

    @lru_cache
    def pca_weight_normalized(self, j, key, w_epoch=None, c_epoch=None):
        """ Returns the normalized weight of a given exp in PCA space, as a (N, R) tensor. """
        kwargs = dict(j=j, key=key, w_epoch=w_epoch, c_epoch=c_epoch)
        return self.pca_weight(**kwargs) / self.pca_weight_norm(**kwargs)  # (N, R)

    @lru_cache
    def whitened_weights(self, j, cat=True, epoch=None):
        """ Returns all the whitened weights in PCA space, as a (num_exps, N, R) or (N_tot, R) tensor. """
        pca_weights = self.aligned_weights(j=j, cat=cat, epoch=epoch) @ self.covariance(j=j, epoch=epoch).eigenvectors.T   # (*, R)
        white_weights = pca_weights / torch.sqrt(self.covariance(j=j, epoch=epoch).eigenvalues)  # (*, R)
        return white_weights

    @lru_cache
    def mean_classifier(self, epoch=None, align_epoch=None) -> torch.Tensor:
        return torch.mean(self.aligned_weights(j=self.exp.num_layers, cat=False,
                                               epoch=epoch, align_epoch=align_epoch), dim=0)  # (C, D)

    def data_covariance(self, j, key=None):
        """ Returns data covariance as a (D_key, D_key) DecomposedMatrix (key defaults to mean data covariance). """
        if key is None:
            key = self.mean_key
        return self.data_covariances[key, j]


def resample(aligned_exps: Dict[Any, AlignedExperiments], resamples: Dict[Any, Tuple[Any, str]], js=None,
             batch_size=1024, align_test=False, total_classes=1000, sub_classes=None, epoch=None):
    """ Resample a given network or ensemble of networks.
    :param aligned_exps: dictionary of align_key -> AlignedExperiments, used for the covariances and the alignment
    :param js: list of layers to resample (can include the classifier)
    :param resamples: dictionary of resample_key -> (align_key_rainbow, align_key_trained, method, new_n)
    - align_key_rainbow is the covariance + mean network to use
    - align_key_trained is the trained network to resample (defines number of atoms)
    - methods is "orthogonal" or "gaussian", determines the type of random matrix that we use with the covariance
    - new_n is a dictionary j -> new_n to use (or None). Unfortunately does not work because state_dict requires same shape...
    :param align_test: whether to align on test set
    :param sub_classes: possibly restrict the number of classes
    :param epoch: possibly use a given epoch for computations (model weights, alignments are computed at this epoch)
    :param return: resampled models, checkpoint_dicts, results (resample_key, metric, j) -> performance (scalar tensor)
    """
    # We need the first layer to be in js, otherwise the assumption of alignment of the first resampled layer does not hold.

    all_exps = {}  # all experiments (models) used across all resamplings
    # This relies on using full names for experiments, so that duplicates can be identified.
    for aligned_name, exps in aligned_exps.items():
        all_exps.update(exps.exps_dict)
        # Disabled prefix behavior:
#         for exp_name, exp in exps.exps.items():
#             all_exps[f"{aligned_name}-{exp_name}"] = exp

    exp = list(all_exps.values())[0]
    if js is None:
        js = exp.js
    align_dataloader = exp.dataloaders(batch_size, total_classes=total_classes,
                                       sub_classes=sub_classes)[1 if align_test else 0]
    conv_shapes = exp.conv_spatial_shapes
    linear_shapes = exp.linear_spatial_shapes

    # All models that we need to compute for alignments purposes (will be populated with new models as well).
    all_models = {key: exp.model(epoch=epoch, cuda=True, merge_std=False) for key, exp in all_exps.items()}
    # Placeholder state dicts for the new models.
    state_dicts = {resample_key: aligned_exps[align_key_trained].exp.model(epoch=epoch, cuda=True, merge_std=False).state_dict().copy()
                   for resample_key, (_, align_key_trained,_, _) in resamples.items()}
    # At beginning of loop, this contains alignment at layer j between resampled model and mean/ref model
    # (used to determine the correct covariance to resample atoms at layer j).
    # TODO
    alignments = {resample_key: aligned_exps[aligned_key].alignment(j=js[0], key=aligned_exps[aligned_key].key, epoch=epoch)
                  for resample_key, (aligned_key, _, _) in resamples.items()}
    # resample_key -> alignment matrix (D_mean, D_resample)

    results = {}  # resample_key, metric, j -> perf

    for i, j in enumerate(js):
        # First build intermediate models, with a new p_j but p_j+1 not aligned yet.

        for resample_key, (align_key_rainbow, align_key_trained, method, new_n) in resamples.items():
            exps = aligned_exps[align_key_rainbow]
            exp = aligned_exps[align_key_trained].exp
            if new_n is None:
                new_n = {}

            if j < exp.num_layers:
                # Resample a P using the aligned covariance.
                covariance = exps.covariance(j=j, epoch=epoch, align_epoch=epoch)  # (D_mean, D_mean)
                n = new_n.get(j, exp.num_atoms(j=j))
                d = exp.dimension(j=j)

                if method == "orthogonal":
                    cov_lean_sqrt = torch.sqrt(covariance.eigenvalues[:, None]) * covariance.eigenvectors  # (R, D_mean)
                    random = linalg.random_orthogonal((n, covariance.rank), device=covariance.matrix.device)  # (N, R)
                    new_p = np.sqrt(n) * random @ cov_lean_sqrt  # (N, D_mean)
                elif method == "gaussian":
                    new_p = linalg.random_gaussian((n, d), cov=covariance)  # (N, D_mean)
                else:
                    raise ValueError(f"Unknown resampling method: {method}")

                # Realign the P from mean space to resampled space.
                alignment = alignments[resample_key]  # (D_mean, D_resample)
                new_p = exps.align_weight(weight=new_p, alignment=alignment, j=j)  # (N, D_resample)

                if j in conv_shapes:
                    new_p = new_p.reshape((n, -1) + conv_shapes[j])  # (N, D_resample, k, k)
            else:
                # Use the aligned mean classifier.
                classifier = exps.mean_classifier(epoch=epoch, align_epoch=epoch)  # (C, D_meanMN)
                alignment = alignments[resample_key]  # (D_mean, D_resample)
                new_p = exps.align_weight(weight=classifier, alignment=alignment, j=j)  # (C, D_resampleMN)

            old_p = state_dicts[resample_key][exp.p_param_name(j=j)]
            print(f"old_p {tensor_summary_stats(old_p)} new_p {tensor_summary_stats(new_p)}")
            state_dicts[resample_key][exp.p_param_name(j=j)] = new_p
            all_models[resample_key] = exp.model_with_state(state_dict=state_dicts[resample_key], ignore_errors=True)

        # Then compute alignment of each resampled network with the corresponding means.

        if j < exp.num_layers:
            next_j = js[i + 1]  # This is different from j + 1 when there are identity P_j's.

            def all_mean_networks(output):
                means = {}
                for align_key, exps in aligned_exps.items():
                    means.update(exps.mean_network(mean_key=f"{align_key}-mean")(output))
                return means

            pairs = [(f"{align_key_rainbow}-mean", resample_key)
                     for resample_key, (align_key_rainbow, _, _, _) in resamples.items()]
            covs = pairwise_model_covariances(all_models, pairs=pairs, additional=all_mean_networks, shapes=linear_shapes,
                                              dataloader=align_dataloader, js=[next_j], compute_std=True)


            for resample_key, (align_key_rainbow, align_key_trained, _, _) in resamples.items():
                exps = aligned_exps[align_key_rainbow]
                exp = aligned_exps[align_key_trained].exp

                cov = covs[f"{align_key_rainbow}-mean", resample_key, next_j]  # (D_mean, D_resample)

                if exp.mean_param_name(j=next_j) is not None:
                    # Recompute the correct stdandardization parameters new_mean and new_var.
                    # We have the relations m1 = (new_mean - old_mean) / sqrt(old_var)
                    # and m2 = new_var / old_var + m_1^2.
                    # We account for the fact that standardization layers add eps to the variance.
                    eps = 1e-5
                    old_mean = state_dicts[resample_key][exp.mean_param_name(j=next_j)]  # (C,), mean before resampling
                    if old_mean.ndim == 2:
                        old_mean = torch.view_as_complex(old_mean)
                    old_var = state_dicts[resample_key][exp.var_param_name(j=next_j)] + eps  # (C,), var before resampling
                    m1 = covs[resample_key, next_j, 1]  # (C,), first moment of representation
                    m2 = covs[resample_key, next_j, 2]  # (C,), second moment of representation
                    new_mean = torch.sqrt(old_var) * m1 + old_mean  # (C,)
                    if torch.is_complex(new_mean):
                        new_mean = torch.view_as_real(new_mean)
                    new_var = old_var * (m2 - torch.abs(m1) ** 2)
                    state_dicts[resample_key][exp.mean_param_name(j=next_j)] = new_mean
                    state_dicts[resample_key][exp.var_param_name(j=next_j)] = new_var

                    # Also recompute the correct covariance (and thus alignment) matrices.
                    # We have old_cov = new_cov * sqrt(old_var / new_var)
                    # Idem, standardization layers will add eps to the variance
                    old_cov = cov.matrix  # (D_mean, D_resample)
                    new_cov = old_cov * torch.sqrt(old_var / (new_var + eps))[None, :]  # (D_mean, D_resample)
                    cov = linalg.DecomposedMatrix(new_cov, decomposition="svd")  # (D_mean, D_resample)

#                     from utils import tensor_summary_stats
#                     tss = lambda t: f"{tensor_summary_stats(t, '.5f')}"
#                     print(f"old_mean {tss(old_mean)}\nnew_mean {tss(new_mean)}\nold_var {tss(old_var)}\nnew_var {tss(new_var)}")

                    variance_resampled = torch.ones((), dtype=cov.dtype, device=cov.device)
                else:
                    variance_resampled = torch.mean(covs[resample_key, next_j, 2])
                                                    # - covs[resample_key, next_j, 1] ** 2)
                # Let's report alignment performance.
                variance_original = torch.mean(covs[f"{aligned_key}-mean", next_j, 2])
                                               # - covs[f"{aligned_key}-mean", next_j, 1] ** 2)
                variance_explained = torch.mean(cov.eigenvalues)
                unexplained_fraction = (variance_original + variance_resampled - 2 * variance_explained) / (2 * variance_original)

                print(f"Original variance {variance_original.item():.2e} resampled {variance_resampled.item():.2e} explained {variance_explained.item():.2e} unexplained fraction {100 * unexplained_fraction:.2f}%")

                alignment = cov.orthogonalize().matrix  # (D_mean, D_resample)
                alignments[resample_key] = alignment  # (D_mean, D_resample)

                # Temporarily, align p_j+1 in models (going to be overriden, but useful for debugging purposes).
                # TODO
                old_p_next = exps.aligned_weight(j=next_j, key=exps.key, epoch=epoch, align_epoch=epoch)  # (N, D_mean)
                new_p_next = exps.align_weight(weight=old_p_next, alignment=alignment, j=next_j)  # (N, D_resample)
                if next_j in conv_shapes:
                    new_p_next = new_p_next.reshape((new_p_next.shape[0], -1) + conv_shapes[next_j])  # (N, D_resample, k, k)

                state_dicts[resample_key][exp.p_param_name(j=next_j)] = new_p_next
                all_models[resample_key] = exp.model_with_state(state_dict=state_dicts[resample_key], ignore_errors=True)

        # Evaluate models.

        for eval_dataloader in exp.dataloaders(batch_size, total_classes=total_classes, sub_classes=sub_classes)[1:]:
            results_j = evaluate(all_models, eval_dataloader)
            print(f"j={j}")
            for (model, metric), acc in results_j.items():
                results[model, metric, j] = acc
                if metric in ["top1", "top5"]:
                    print(f"{model}: {acc.item():.1f}% {metric}")

    # Filter the all_models dict to return only the resampled models.
    resampled_models = {resample_key: all_models[resample_key] for resample_key in resamples}
    checkpoint_dicts = {resample_key: dict(
            epoch=0,
            best_acc1=0,
            state_dict=state_dicts[resample_key],
        ) for resample_key in resamples}
    return resampled_models, checkpoint_dicts, results


def clip(aligned_exps: Dict[Any, AlignedExperiments], js, clips: Dict[Any, Tuple[Any, str]],
         batch_size=1024, merge_std=False):
    """ Clip a given network or ensemble of networks.
    :param aligned_exps: dictionary of align_key -> (AlignedExperiments, epoch), used for the possible alignment
    :param js: list of layers to clip
    :param clips: dictionary of clip_key -> (align_key, clip_dimensions)
    - align_key is the reference network to use (plus eventual realignment)
    - clip_dimensions says for each j the number of eigenvectors to keep
    :return: all_models, results
    - all_models is a dictionary of align_key -> clipped model
    - results is a dictionary of align_key, "top{k}" -> accuracy
    """
    all_exps = {}  # all experiments (reference models) used across all clippings
    # This relies on using full names for experiments, so that duplicates can be identified.
    for aligned_name, (exps, epoch) in aligned_exps.items():
        all_exps[aligned_name] = (exps.exp, epoch)

    exp = list(all_exps.values())[0][0]
    dataloader = exp.train_dataloader(batch_size)
    shapes = exp.linear_spatial_shapes

    # All models that we need to compute for comparing performance (will be populated with new models as well).
    all_models = {key: exp.model(epoch=epoch, cuda=True, merge_std=merge_std) for key, (exp, epoch) in all_exps.items()}
    # Placeholder state dicts for the new models.
    state_dicts = {clip_key: aligned_exps[aligned_key][0].exp.model(epoch=None, cuda=True, merge_std=merge_std).state_dict().copy()
                   for clip_key, (aligned_key, _) in clips.items()}
    alignments = {}  # clip_key -> alignment matrix (D_mean, D_resample)

    for i, j in enumerate(js):
        # First build intermediate models, with a clipped p_j but p_j+1 not aligned yet.

        for clip_key, (aligned_key, dimensions) in clips.items():
            exps = aligned_exps[aligned_key][0]
            exp = exps.exp

            n = exp.num_atoms(j=j)  # We could imagine specifying new widths
            d = exp.dimension(j=j)
            projector = exps.covariance(j=j).project(dimensions[i]).orthogonalize().matrix  # (D_mean, D_mean)

            old_p = state_dicts[clip_key][exp.p_param_name(j=j)][:, :, 0, 0]  # (N, D)
            alignment = exps.mean_alignments[exps.key, j]  # (D, D_mean)
            new_p = old_p @ alignment @ projector @ alignment.T  # (N, D)

            state_dicts[clip_key][exp.p_param_name(j=j)] = new_p[:, :, None, None]
            all_models[clip_key] = exp.model_with_state(state_dict=state_dicts[clip_key])

        # Then compute alignment of each resampled network with the corresponding means.

#         if j < exp.num_layers:
#             next_j = js[i + 1]  # This is different from j + 1 when there are identity P_j's.

#             def all_mean_networks(output):
#                 means = {}
#                 for aligned_key, exps in aligned_exps.items():
#                     means.update(exps.mean_network(mean_key=f"{aligned_key}-mean")(output))
#                 return means

#             pairs = [(f"{aligned_key}-mean", resample_key)
#                      for resample_key, (aligned_key, method) in resamples.items()]
#             covs = pairwise_model_covariances(all_models, pairs=pairs, additional=all_mean_networks, shapes=shapes,
#                                               dataloader=dataloader, js=[next_j], compute_std=True)

#             for resample_key, (aligned_key, method) in resamples.items():
#                 exps = aligned_exps[aligned_key]
#                 exp = exps.exp

#                 cov = covs[f"{aligned_key}-mean", resample_key, next_j].matrix  # (D_mean, D_resample)

#                 if "Std" in exp.args.arch:
#                     # Recompute the correct stdandardization parameters new_mean and new_var.
#                     # We have the relations m1 = (new_mean - old_mean) / sqrt(old_var)
#                     # and m2 = new_var / old_var + m_1^2.
#                     old_mean = state_dicts[resample_key][exp.mean_param_name(j=next_j)]  # (C,), mean before resampling
#                     old_var = state_dicts[resample_key][exp.var_param_name(j=next_j)]  # (C,), var before resampling
#                     m1 = covs[resample_key, next_j, 1]  # (C,), first moment of representation
#                     m2 = covs[resample_key, next_j, 2]  # (C,), second moment of representation
#                     new_mean = torch.sqrt(old_var) * m1 + old_mean  # (C,)
#                     new_var = old_var * (m2 - m1 ** 2)
#                     state_dicts[resample_key][exp.mean_param_name(j=next_j)] = new_mean
#                     state_dicts[resample_key][exp.var_param_name(j=next_j)] = new_var

#                     # Also recompute the correct covariance (and thus alignment) matrices.
#                     # We have old_cov = new_cov * sqrt(old_var / new_var)
#                     old_cov = cov  # (D_mean, D_resample)
#                     new_cov = old_cov * torch.sqrt(old_var / new_var)[None, :]  # (D_mean, D_resample)
#                     cov = new_cov  # (D_mean, D_resample)

#                 alignment = linalg.orthogonalize(cov)  # (D_mean, D_resample)
#                 alignments[resample_key] = alignment  # (D_mean, D_resample)

#                 # Temporarily, align p_j+1 in models (going to be overriden, but useful for debugging purposes).
#                 old_p_next = exps.aligned_weight(j=next_j, key=exps.key, epoch=None)  # (N, D_mean)
#                 new_p_next = exps.align_weight(weight=old_p_next, alignment=alignment, j=next_j)  # (N, D_resample)
#                 if next_j < exp.num_layers:
#                     new_p_next = new_p_next[:, :, None, None]

#                 state_dicts[resample_key][exp.p_param_name(j=next_j)] = new_p_next
#                 all_models[resample_key] = exp.model_with_state(state_dict=state_dicts[resample_key])

        # Evaluate models.

        for dataloader in exp.dataloaders(batch_size)[1:]:
            results = evaluate(all_models, dataloader)
            print(f"j={j}")
            for (model, metric), acc in results.items():
                if metric == "top1":
                    print(f"{model}: {acc.item():.1f}%")

    return all_models, results


def evaluate(models: Dict[Any, torch.nn.Module], dataloader) -> Dict[Tuple[Any, str], float]:
    """ Returns Top1 and Top5 accuracy of several models.
    :param models: dictionary of models to evaluate
    :param dataloader: dataloader to evaluate on
    :return: dictionary model_key, "top{k}" -> top-k accuracy of given model
    """
    ks = (1, 5)

    def closure(x, y):
        outputs = {}
        for key, model in models.items():
            for k, acc in zip(ks, main_block.accuracy(model(x), y, topk=ks)):
                # Slightly ugly hack to have a (B,) tensor for expected_value.
                outputs[key, f"top{k}"] = acc * torch.ones((x.shape[0],), device=x.device)
        return outputs

    return expected_value(closure, dataloader)


def gaussian_dataloader(num_samples=50000, image_size=32, grayscale=True, c=0.05, alpha=2, batch_size=128):
    """ Returns a Gaussian dataloader, with a circular power spectrum proportional to 1 / (c + ||om||^alpha).
    Data is zero-mean and unit-variance. """
    device = torch.device("cpu")  # Loaded to GPU per-batch anyway.

    # Compute normalized spectrum.
    x, y = torch.meshgrid(*(
        N * torch.fft.fftfreq(N, device=device) for N in (image_size,) * 2
    ), indexing="ij")  # (H, W) both
    omega_norm = torch.sqrt(x ** 2 + y ** 2)  # (H, W)
    spectrum = 1 / (c + omega_norm ** alpha)
    spectrum /= spectrum.mean()

    # Sample from a stationary Gaussian in Fourier domain.
    shape = (num_samples, 1 if grayscale else 3, image_size, image_size, 2)
    noise_fft = torch.view_as_complex(torch.randn(shape, device=device))  # (*, C, H, W) complex
    # E[|noise_fft|²] = 2 for now, so we rescale to have the right spectrum.
    # Note: we keep this factor of sqrt(2) because we discard the imaginary part after the IFFT.
    noise_fft *= torch.sqrt(spectrum)
    # Go to space domain, discarding the imaginary part.
    noise = torch.real(torch.fft.ifft2(noise_fft, norm="ortho"))  # (*, C, H, W) real

    # Create dataset and dataloader.
    dataset = torch.utils.data.TensorDataset(noise, torch.zeros((num_samples,), device=device))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return dataloader


exps = Experiments()  # Global experiments object.

def exp_pattern(dataset="cifar", base_sizes=[64, 128, 256, 512], js=[1, 2, 3], s=1, std=True, L=4, name=None,
                intermediate="clbnstd-clnobias-nowd-paperexps"):
    """ Standard pattern for experiments. """
    sizes = [int(size * s) if j in js else size for j, size in enumerate(base_sizes, start=1)]
    arch = "W" + "W".join(f"P{size}" if size != "id" else "" for size in sizes) + (f"-L{L}" if std else "-nostd")
    return f"{dataset}-{arch}-{intermediate}-{name + '-' if name is not None else ''}init"
