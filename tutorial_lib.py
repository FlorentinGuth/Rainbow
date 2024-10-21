import time
from tqdm.notebook import tqdm
import copy

import torch
from torch import nn
import torch.utils.data.dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_datasets():
    """ Returns train and validation datasets. """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std), transforms.Grayscale()]

    dataset_class = datasets.CIFAR10
    def get_dataset(train: bool):  # Returns the train or validation dataset.
        root = "./data"
        kwargs = dict(root=root, transform=transforms.Compose(data_transforms), train=train, download=True)
        dataset = dataset_class(**kwargs)
        return dataset

    train_dataset = get_dataset(train=True)
    val_dataset = get_dataset(train=False)
    return train_dataset, val_dataset


def get_dataloaders(batch_size):
    """ Returns train and validation dataloaders. """
    train_dataset, val_dataset = get_datasets()

    def get_dataloader(dataset, shuffle):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                           num_workers=0, pin_memory=True)

    train_loader = get_dataloader(train_dataset, shuffle=True)
    val_loader = get_dataloader(val_dataset, shuffle=False)
    return train_loader, val_loader


def initialize_mlp_model(dims):
    """ Create a simple MLP model.
    :param dims: list of dimensions of each layer, should begin with dimension of input and end with number of classes.
    :return: Sequential MLP model
    """
    layers = [torch.nn.Flatten()]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    return model


def initialize_cnn_model(channels, spatial_size=32, kernel_size=5, stride=2, num_classes=10):
    """ Create a simple CNN model.
    :param channels: list of channels of each convolutional layer, should begin with number of channels of input.
    :return Sequential CNN model
    """
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=5, stride=2, bias=False))
        spatial_size = ceil_div(spatial_size - kernel_size + 1, stride)
        layers.append(nn.ReLU())

    layers.extend([nn.Flatten(), nn.Linear(channels[-1] * spatial_size ** 2, num_classes, bias=False)])
    model = nn.Sequential(*layers)
    return model


def ceil_div(a: int, b: int) -> int:
    """ Return ceil(a / b). """
    return a // b + (a % b > 0)


def train_model(model, train_loader, val_loader, lr=0.01, momentum=0.9, num_epochs=5):
    """ Simple training of a model with SGD. """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    epoch = 0
    model.cuda()
    while True:
        # Evaluate on validation set.
        one_epoch(
            loader=val_loader, model=model, loss_fn=loss_fn, 
            optimizer=optimizer, epoch=epoch, is_training=False,
        )

        # Stop if we are at the last epoch.
        if epoch == num_epochs:
            break

        # Train for one epoch (now epoch counts the current training epoch).
        epoch += 1
        one_epoch(
            loader=train_loader, model=model, loss_fn=loss_fn, 
            optimizer=optimizer, epoch=epoch, is_training=True,
        )
        
    # Ensure evaluation mode and disable gradients before returning trained model.
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def one_epoch(loader, model, loss_fn, optimizer, epoch, is_training):
    """ Run one train or validation epoch on the model.
    :param loader: dataloader to use
    :param model: model to train or evaluate
    :param loss_fn: loss function (not used during evaluation)
    :param optimizer: optimizer (not used during evaluation)
    :param epoch: current epoch number (for tqdm description)
    :param is_training: whether to train the model or simply evaluate it
    :return: average accuracy during the epoch
    """
    name_epoch = "Train" if is_training else "Val"
    name_epoch = f"{name_epoch} epoch {epoch}"
    accuracy_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_training):
        it = tqdm(loader, desc=name_epoch)
        for x, y in it:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            y_hat = model(x)

            loss = loss_fn(y_hat, y)
            accuracy = 100 * torch.sum((y_hat.argmax(-1) == y).float())
            accuracy_meter.update(val=accuracy.item(), n=y.shape[0])
            it.desc = f"{name_epoch}: {accuracy_meter.avg():.2f}% accuracy"

            if is_training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    return accuracy_meter.avg()
    
    
class AverageMeter(object):
    """ Computes and stores the average and current value. """
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val
        self.count += n
        
    def avg(self):
        return self.sum / self.count

    
def get_activations(x, layers, model):
    """ Returns the hidden activations of a model.
    :param x: input to use, tensor of shape (B, C, [N, N])
    :param layers: list of integers (j corresponds to output of j-th layer, 0 corresponds to input of model)
    :param model: model to use (should be Sequential)
    :return: list of saved activations of same length as layers
    """
    saved_activations = []

    def hook(self, inputs, output):  # inputs is a tuple, we assume it is of length 1
        saved_activations.append(inputs[0])
    
    # Register hooks to save activations of chosen layers.
    for layer in layers:
        model[layer].register_forward_hook(hook)

    # Forward of model: hooks will be run and activations will be saved.
    _ = model(x)
    
    # Clear hooks.
    for layer in layers:
        model[layer]._forward_hooks.clear()
        
    return saved_activations


def compute_activation_covariances(loader, layers, model1, model2=None):
    """ Compute the (cross-)covariance of hidden activations at several layers of one or two models.
    :param loader: data loader to use
    :param layers: list of integers (j corresponds to output of j-th layer, 0 corresponds to input of model)
    :param model1: model to use (should be Sequential)
    :param model2: optional model for a cross-covariance (if None, compute the self-covariance of model1)
    :return: list of computed covariances (C1, C2), of same length as layers
    """
    meters = [AverageMeter() for _ in layers]
    
    # Version of get_activations which treats spatial dimensions as additional batch dimensions.
    get_acts = lambda *args: [space_to_batch(act) for act in get_activations(*args)]
    
    for x, _ in tqdm(loader, desc="Computing activation covariance"):
        x = x.cuda()
        activations1 = get_acts(x, layers, model1)
        activations2 = activations1 if model2 is None else get_acts(x, layers, model2)
        
        for i, (act1, act2) in enumerate(zip(activations1, activations2)):
            cov = act1.T @ act2  # (C1, C2), sum of outer products over the batch
            meters[i].update(val=cov, n=act1.shape[0])
    
    return [meter.avg() for meter in meters]


def space_to_batch(x):
    """ (B, C, [M, N]) to (B[MN], C). """
    if x.ndim == 4:
        x = x.permute(0, 2, 3, 1)  # (B, M, N, C)
        x = x.reshape((-1, x.shape[-1]))  # (BMN, C)
    return x


def clip(model, layer, basis, dimensions, loader):
    """ Clip and evaluate a model.
    :param model: trained model to clip (should be Sequential)
    :param layer: index of layer to clip
    :param basis: ordered orthogonal basis vectors for the clipping, of shape (num_vectors, input_dim)
    :param dimensions: list of clipping dimensions
    :param loader: data loader to use for the evaluation
    :return: list of accuracies
    """
    models = []
    for dim in dimensions:
        projector = basis[:dim].T @ basis[:dim]
        model_clipped = copy.deepcopy(model)
        model_clipped[layer].weight = torch.nn.Parameter(model_clipped[layer].weight @ projector, requires_grad=False)
        models.append(model_clipped)
    
    return evaluate(models, loader, desc="Evaluation after clipping")


def evaluate(models, loader, desc="Evaluation"):
    """ Evaluate a list of models.
    :param models: list of models to evaluate
    :param loader: dataloader to use for evaluation
    :return: list of accuracies, one per model
    """
    accuracy_meters = [AverageMeter() for _ in models]

    it = tqdm(loader, desc=desc)
    for x, y in it:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        for model, accuracy_meter in zip(models, accuracy_meters):
            y_hat = model(x)
            accuracy = 100 * torch.sum((y_hat.argmax(-1) == y).float())
            accuracy_meter.update(val=accuracy.item(), n=y.shape[0])
    
    return [accuracy_meter.avg() for accuracy_meter in accuracy_meters]


def resample(reference_model, state_dict_keys, train_loader, val_loader): 
    """ Resample a given network or ensemble of networks.
    NOTE: this function does not take care of batch norm parameters. 
    They should be recomputed when the previous layer has been resampled.
    NOTE: this function can be extended in many ways: resample only a subset of layers, change the width of some layers,
    use weight covariances averaged over several models...
    :param reference_model: reference model to resample
    :param state_dict_keys: list of keys in model state dict to resample
    :param train_loader: dataloader used for computing alignment between reference and resampled model
    :param val_loader: dataloader used to evaluate performance of resampled model
    :return: list of accuracies, corresponding to the reference and resampled models
    """
    # The resampled model is initialized as a copy of the reference model.
    resampled_model = copy.deepcopy(reference_model)
    state_dict = resampled_model.state_dict()
    
    # At beginning of loop, this contains alignment at layer j between resampled model and reference model
    # (used to determine the correct covariance to resample atoms at layer j).
    alignment = None  # alignment matrix (C_in_reference, C_in_resample)
    
    results = {}  # metric, layer_idx -> performance
    
    # Resample each layer iteratively.
    for j, key in enumerate(state_dict_keys):
        weight = state_dict[key]  # (C_out, C_in, [h, w]) depending on conv or fully connected layer
        
        # Compute new weight in reference space.
        if j < len(state_dict_keys) - 1:
            # Sample Gaussian weights with the same covariance as the trained weights.
            # Compute square root of covariance with an SVD (more efficient when C_out << C_in*h*w).
            weight_flat = weight.reshape((weight.shape[0], -1))  # (C_out, C_in*h*w)
            u, s, vt = torch.linalg.svd(weight_flat, full_matrices=False)  # (C_out, R), (R,), (R, C_in*h*w) where R = rank
            white_gaussian = torch.randn(u.shape, dtype=weight.dtype, device=weight.device)  # (C_out, R)
            colored_gaussian = white_gaussian @ (s[:, None] * vt)  # (C_out, C_in*h*w)
            new_weight = colored_gaussian.reshape(weight.shape)  # (C_out, C_in, [h, w])
        else:
            # Use the trained classifier.
            new_weight = weight
        
        # Realign the weights from the reference model to the resampled model (necessary after first layer).
        if j > 0:
            new_weight = contract(new_weight, alignment.T, axis=1)  # C_in_reference to C_in_resampled

        # Set the new weights in the resampled model.
        # NOTE: this an intermediate model, as resampling the j-th layer means that the j+1-th layer is no longer aligned.
        # As such, if evaluated as is, its accuracy would be that of a random guess.
        state_dict[key] = new_weight
        resampled_model.load_state_dict(state_dict)
        
        # Then compute alignment of the resampled network with the reference model for the next layer.
        if j < len(state_dict_keys) - 1:
            next_key = state_dict_keys[j + 1]
            # Compute index of layer by relying on Sequential naming convention: next_key is "{layer_idx}.weight".
            layer = int(next_key.split(".")[0])
            
            [activation_covariance] = compute_activation_covariances(
                train_loader, [layer], reference_model, resampled_model)  # (C_in_reference, C_in_resampled)

            u, s, vh = torch.linalg.svd(activation_covariance, full_matrices=False)  # (C_in_reference, R), (R,), (R, C_in_resampled)
            alignment = u @ vh  # (C_in_reference, C_in_resampled)

    # Evaluate models.
    return evaluate([reference_model, resampled_model], val_loader, desc="Evaluation after resampling")


def contract(tensor, matrix, axis):
    """ tensor is (..., D, ...), matrix is (P, D), returns (..., P, ...). """
    t = torch.moveaxis(tensor, source=axis, destination=-1)  # (..., D)
    r = t @ matrix.T  # (..., P)
    return torch.moveaxis(r, source=-1, destination=axis)  # (..., P, ...)
