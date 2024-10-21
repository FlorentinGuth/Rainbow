""" Handles any model-related operations: loading, analyzing, etc. """

import torch
import numpy as np
import time
from typing import *

import main_block
from models.LinearProj import ComplexConv2d, TriangularComplexConv2d
from models.Classifier import Classifier
import utils

utils.do_svd = False  # Disable SVD on model initialization (on CPU...) because we resume from checkpoints.


def get_state_dict(checkpoint_file, cuda=False, log=False):
    # Three tries because the experiment might be running and writing to the file:
    num_tries = 3
    for i in range(num_tries):
        try:
            checkpoint_dict = torch.load(str(checkpoint_file), map_location=torch.device('cuda:0' if cuda else 'cpu'))
            break
        except Exception as e:
            if i == num_tries:
                raise e
            else:
                print(f"Try {i+1}/{num_tries}: Exception {e} ignored, waiting and retrying")
                time.sleep(5)
    if log:
        print(f"Got model at epoch {checkpoint_dict['epoch']} with best acc1 {checkpoint_dict['best_acc1']}%")
    return checkpoint_dict["state_dict"]


def get_model(args, state_dict, cuda=False, log=False, ignore_errors=False):
    """ Returns the pytorch model with trained weights ready for evaluation. """
    model = main_block.load_model(args, logfile=None, summaryfile=None, writer=None, log=log)

    # Loads model weights, ignoring key mismatches (they are printed as warnings).                
    k1 = list(state_dict.keys())
    k2 = list(model.state_dict().keys())
    for k in k1:
        if k not in k2:
            print(utils.red_color(f"Checkpoint has extra key {k}"))
            del state_dict[k]
    for k in k2:
        if k not in k1:
            print(utils.red_color(f"Model has extra key {k}"))
            state_dict[k] = model.state_dict()[k]
            
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as err:
        if ignore_errors:
            print(f"Ignored load_state_dict errors: {err}")
        else:
            raise err
    
    # Final touch: put model in evaluation mode and disable all gradients.
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if cuda:
        model.cuda()  # For some reason this is necessary.
    
    if log:
        print(f"Loaded model {args.dir}")

    return model


def get_conv_modules(model):
    """ Returns a list of ComplexConv2D modules found in model. Convolutions with zero-sized output are ignored. """
    convs = []
    for module in model.modules():
        if any(isinstance(module, conv) for conv in [torch.nn.Conv2d, ComplexConv2d, TriangularComplexConv2d]):
            if module.out_channels > 0:
                convs.append(module)
    return convs


def get_linear_modules(model):
    """ Return a list of Linear modules found in model. """
    linears = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            linears.append(module)
    return linears
        
        
def get_classifier(model):
    """ Return the Classifier module found in model (scattering network only). """
    for module in model.modules():
        if isinstance(module, Classifier):
            return module
        
        
def get_weight_modules(model):
    """ Returns a list of convolution modules + linear modules. """
    return get_conv_modules(model) + get_linear_modules(model)


def get_weight_matrix(module):
    """ Returns the weights of the module as a real or complex 2-dimensional tensor. """
    if isinstance(module, torch.nn.Linear):
        return module.weight
    
    # module is a convolution.
    if isinstance(module, torch.nn.Conv2d):
        w = module.weight
    elif hasattr(module, "full_weights"):
        w = module.full_weights()
    else:
        w = module.param
        
    # w is (C_out, C_in, k_x, k_y, [2]) depending on real or complex weights.
    # Reshape to (C_out, C_in*k_x*k_y) and view as complex if necessary.
    assert w.ndim in [4, 5]
    w = w.data.reshape(w.shape[:1] + (-1,) + w.shape[4:])
    if w.ndim == 3:
        w = torch.view_as_complex(w)
    return w


def reshape_atoms(P, N_prev=None, L=None, axis=1):
    """ Rearrange weights from (*, N_prev*(1+L), *) to (*, N_prev, 1+L, *). """
    
    def index(i):
        if not isinstance(i, tuple):
            i = (i,)
        return (slice(None,),) * axis + i
    
    def reshape(x, s):
        return x.reshape(x.shape[:axis] + s + x.shape[axis+1:])
    
    if N_prev is None:
        assert P.shape[axis] % (1 + L) == 0
        N_prev = P.shape[axis] // (1 + L)
    
    P_phi = P[index((slice(None, N_prev), None))]  # (*, N_prev, 1, *)
    P_psi = reshape(P[index(slice(N_prev, None))], (N_prev, -1))  # (*, N_prev, L, *)
    return torch.cat((P_phi, P_psi), dim=axis + 1)


def get_activations(x: torch.Tensor, model: torch.nn.Module, to_save: Dict[str, Tuple[str, torch.nn.Module]]) -> torch.Tensor:
    """ Computes the inputs and outputs of a list of modules.
    :param x: input to use, tensor of shape (B, C, N, N)
    :param model:
    :param to_save: dictionary of key -> ("input"/"output", module)
    :return: saved activations, dictionary of key -> (B, Cj, Nj, Nj) (input or output of corresponding module)
    """
    saved_activations = {}

    def get_hook(key, input_output):
        def hook(self, inputs, output):  # inputs is a tuple, we assume it is of length 1
            if input_output == "input":
                saved_activations[key] = inputs[0]
            elif input_output == "output":
                saved_activations[key] = output
            return output
        return hook
    
    handles = []
    for key, (input_output, module) in to_save.items():
        handles.append(module.register_forward_hook(get_hook(key, input_output)))
    model(x)
    
    # Remove hooks: handles do not seem to work?
    for key, (_, module) in to_save.items():
        module._forward_hooks.clear()
        
    return saved_activations


def get_grad(x: torch.Tensor, model: torch.nn.Module, module: torch.nn.Module) -> torch.Tensor:
    """ Computes the gradient of the predicted class logits with respect to the output of a given module.
    :param x: input to use, tensor of shape (B, C, N, N)
    :param model:
    :param module: the gradient will be computed with respect to the output of this module
    :return: gradient, tensor of shape (B, K, Y),
    where Y is the number of classes and K the number of output channels of module
    """
    leaf = None

    def hook(self, inputs, output):
        nonlocal leaf
        leaf = output  # (B, K, N', N')
        leaf.requires_grad_(True)
        return leaf

    module.register_forward_hook(hook)

    y = model(x).sum(0)  # (Y,)
    grad = torch.stack([torch.autograd.grad(outputs=y[i], inputs=leaf, retain_graph=True)[0]
                        for i in range(len(y))], dim=2)  # (B, K, Y, N, N)
    grad = grad.mean((3, 4))  # (B, K, Y)

    module._forward_hooks.clear()

    return grad
