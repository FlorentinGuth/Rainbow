""" Drop-in replacements for Lucent visualization. """

from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms as transforms

from lucent.optvis import render, param, transform, objectives
from lucent.optvis.param.color import *
from lucent.optvis.param.spatial import *
from lucent.optvis.render import *
from lucent.optvis.transform import *
from lucent.optvis.objectives import *
from lucent.optvis.objectives_util import _extract_act_pos

from utils import SplitTensor


standard_transforms = [
    #     pad(12, mode="constant", constant_value=0.5),
    jitter(2),
    #     random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    #     random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(2),
]


def image(w=32, h=None, sd=None, batch=10, decorrelate=True,
          fft=True, channels=1):
    h = h or w
    batch = batch or 1
    ch = channels or 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd)
    output = to_valid_rgb(image_f, decorrelate=False if ch != 3 else decorrelate)
    return params, output


def fft_image(shape, sd=None, decay_power=1):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,)  # 2 for imaginary and real components
    sd = sd or 0.01

    device = torch.device("cuda:0")
    spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if TORCH_VERSION >= "1.7.0":
            import torch.fft
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        else:
            import torch
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:batch, :channels, :h, :w]
        magic = 4.0  # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        # Disabled because if I ask for grayscale, I want grayscale
        #         if channels==1:
        #           new_shape=copy.deepcopy(shape)
        #           new_shape[1]=3
        #           image = image.expand(new_shape)
        return image

    return [spectrum_real_imag_t], inner


def normalize(channels):
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    if channels == 1:
        normal = transforms.Normalize(mean=[0.481, ], std=[0.239, ])
    elif channels == 3:
        normal = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        assert False

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


def render_vis(
        model,
        objective_f,
        param_f=None,
        optimizer=None,
        transforms=None,
        thresholds=(512,),
        verbose=False,
        preprocess=True,
        progress=True,
        show_image=False,
        save_image=False,
        image_name=None,
        show_inline=False,
        fixed_image_size=None,
):
    """ Returns a (batch, 32, 32) numpy array. """
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            # Use my own normalization for grayscale handling
            transforms.append(normalize(channels=1))

    # Upsample images smaller than 224
    # Disabled because I want to fix the image size
    #     image_shape = image_f().shape
    #     if fixed_image_size is not None:
    #         new_size = fixed_image_size
    #     elif image_shape[2] < 224 or image_shape[3] < 224:
    #         new_size = 224
    #     else:
    #         new_size = None
    #     if new_size:
    #         transforms.append(
    #             torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
    #         )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss: {:.3f}".format(objective_f(hook)))

    images = []
    try:
        for i in tqdm(range(1, max(thresholds) + 1), disable=(not progress)):
            def closure():
                optimizer.zero_grad()
                #                 try:
                image = image_f()
                #                 print(f"Created image of shape {image.shape}")
                image = transform_f(image)
                #                 print(f"Transformed image to shape {image.shape}")
                model(image)
                #                 except RuntimeError as ex:
                #                     if i == 1:
                #                         # Only display the warning message
                #                         # on the first iteration, no need to do that
                #                         # every iteration
                #                         warnings.warn(
                #                             "Some layers could not be computed because the size of the "
                #                             "image is not big enough. It is fine, as long as the non"
                #                             "computed layers are not used in the objective function"
                #                             f"(exception details: '{ex}')"
                #                         )
                loss = objective_f(hook)
                loss.backward()
                return loss

            optimizer.step(closure)
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                if verbose:
                    print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                    if show_inline:
                        show(image)
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        if verbose:
            print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
        images.append(tensor_to_img_array(image_f()))
    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()), width=128)
    elif show_image:
        view(image_f())
    return np.array(images)[0,...,0]


def _T_handle_batch(T, batch=None):
    def T2(name):
        t = T(name)
        
        if isinstance(t, SplitTensor):
            t = t.full_view()
        
        if isinstance(batch, int):
            return t[batch:batch+1]
        else:
            return t
    return T2


def handle_batch(batch=None):
    return lambda obj: lambda model: obj(_T_handle_batch(model, batch=batch))


@wrap_objective()
def neuron(layer, n_channel, x=None, y=None, batch=None):
    """Visualize a single neuron of a single channel.
    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.
    Odd width & height:               Even width & height:
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                      |   |   |   |   |
                                      +---+---+---+---+
    """

    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        return -layer_t[:, n_channel].mean()

    return inner


@wrap_objective()
def channel(layer, n_channels, batch=None):
    """ Visualize a single channel. n_channels can be a list, to optimize simultaneously for different channels. """

    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)  # (B, C, M, N)
        
        if hasattr(n_channels, "__len__"):
            idx = (np.arange(len(n_channels)), n_channels)
            signs = torch.ones(len(n_channels), device=layer_t.device)
        else:
            idx = (slice(None), n_channels)
            signs = torch.cat([torch.ones(layer_t.shape[0]//2, device=layer_t.device), 
                               -torch.ones(layer_t.shape[0] - layer_t.shape[0]//2, device=layer_t.device)])

        return -(signs[:, None, None] * layer_t[idx]).mean()

    return inner


@wrap_objective()
def direction(layer, direction, batch=None):
    """ Visualize a direction.
    Direction is (C, M, N) (include space). """
       
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        
        return -torch.nn.CosineSimilarity(dim=1)(direction[None], layer_t).mean()

    return inner


"""
:param layer: lucent name of layer, retrieve the list of layers with:
              import lucent.modelzoo.util
              lucent.modelzoo.util.get_model_layers(model)
              Example: layer = f"module_{j}_module_3_submodules_dict_(0, 0)"
"""


def visualize_atom(model, obj, batch=10, filename=None, show=True):
    """ Visualizes ten different images which activate the most a given atom.
    :param model: model to visualize
    :param obj: objective to use
    :param batch: number of images to generate
    :param filename: name of file to save images to
    :param show: whether to show the computed images
    :return: numpy array of shape (batch, 32, 32)
    """
    return render_vis(
        model, objective_f=obj, param_f=lambda: image(batch=batch),
        progress=False, show_inline=show, save_image=filename is not None, image_name=filename,
    )
