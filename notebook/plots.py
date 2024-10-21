""" Handles all plotting-related functions. """

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt


from scipy.ndimage import gaussian_filter

from .models import reshape_atoms
from .linalg import *


matplotlib.rcParams.update({
    'text.usetex': True, 
    'font.size': '14',  # Default: 10
    'font.family': 'serif'
})


def to_numpy(data):
    """ Converts a tensor/array/list to numpy array.
    Recurse over dictionaries and tuples. Values are left as-is. """
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(to_numpy(v) for v in data)
    else:
        return data


# General plotting function.

def plot_lines(*lines, x=None, same_x=False, smoothing=None, show_points=True, show_mean=True, show_std=True, subsample_factor=None,
               title=None, xscale="linear", yscale="linear",
               has_fig=False, figsize=None, colormap=None, colorbar=False, colors=None, xlabel=None, ylabel=None, save=None, legend=True, aspect=None,
               xlim=None, ylim=None, alpha=None, xticks=None, xtick_labels=None, yticks=None, ytick_labels=None, marker=None, linestyle=None, linewidth=None, grid=True, **named_lines):
    """ Creates a simple one-off plots with lines.
    :param lines: sequence of unnamed arrays of y-values, assuming x going from 1 to n, or (x, y)
    :param x: optional x to use for all lines (should be the same)
    :param same_x: whether to rescale the x's to [0, 1]
    :param title: title of the plot
    :param colors: list of colors for each line
    :param colormap: assign a color to lines based on its rank
    :param xscale: scale of the x-axis
    :param yscale: scale of the y-axis
    :param has_fig: whether to create/show the plot (useful for subplots)
    :param figsize: size of figure (width, height) in inches, if has_fig is False
    :param named_lines: dictionary of named lines
    """
    
    if not has_fig:
        plt.figure(figsize=figsize)
        
    if isinstance(xscale, str):
        xscale = (xscale, {})
    plt.xscale(xscale[0], **xscale[1])
    if isinstance(yscale, str):
        yscale = (yscale, {})
    scale = matplotlib.scale.scale_factory(yscale[0], plt.gca().yaxis, **yscale[1])
    plt.yscale(yscale[0], **yscale[1])
        
    all_lines = [("", line) for line in lines] + list(named_lines.items())
    all_lines = [(name, to_numpy(line)) for name, line in all_lines]
    
    if not isinstance(linestyle, list):
        linestyle = [linestyle] * len(all_lines)
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * len(all_lines)
    
    if x is not None:
        x = to_numpy(x)

    if colors is None and colormap is not None:
        colors = matplotlib.cm.get_cmap(colormap)(np.linspace(0, 1, len(all_lines)))
        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=matplotlib.colors.Normalize(vmin=1, vmax=len(all_lines)))
            plt.colorbar(sm)

    
    for i, (name, line) in enumerate(all_lines):
        if isinstance(line, tuple):
            this_x, line = line
        elif x is not None:
            this_x = x
        elif same_x:
            this_x = np.linspace(0, 1, len(line))
        else:
            this_x = 1 + np.arange(len(line))
            
        this_color = colors[i] if colors is not None and i < len(colors) else None
        this_linestyle = linestyle[i]
        this_linewidth = linewidth[i]
        
        if smoothing is not None:
            sigma = smoothing * len(line)
            smooth = lambda y: gaussian_filter(y, sigma=sigma, mode="nearest")
            
            line_y = scale.get_transform().transform(line)
            line_mean_y = smooth(line_y)
            line_std_y = np.sqrt(smooth((line_y - line_mean_y) ** 2))
            line_minus, line_mean, line_plus = [scale.get_transform().inverted().transform(line_mean_y + s * line_std_y) for s in [-1, 0, 1]]
            
            if show_mean:
                plotted_line, = plt.plot(this_x, line_mean, label=name, color=this_color, linestyle=this_linestyle, 
                                         linewidth=this_linewidth)
                this_color = plotted_line.get_color()
            if show_std:
                plt.fill_between(this_x, line_minus, line_plus, color=this_color, alpha=0.3)
            if show_points:
                if subsample_factor is not None:
                    this_x = this_x[::subsample_factor]
                    line = line[::subsample_factor]
                plt.scatter(this_x, line, marker="+", color=this_color)
        else:
            plt.plot(this_x, line, label=name, color=this_color, marker=marker, alpha=alpha, 
                     linestyle=this_linestyle, linewidth=this_linewidth)
    
    plt.grid(b=grid, which="both", axis="both")
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
        
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xticks(xticks, xtick_labels)
    plt.yticks(yticks, ytick_labels)
    if title is not None:
        plt.title(title)
    if legend and len(named_lines) > 0:
        plt.legend()
    if aspect is not None:
        plt.gca().set_aspect(aspect)
    plt.tight_layout()
        
    if save is not None:
        savefig(save)
    if not has_fig:
        plt.show()


def plot_img(img, title=None, has_fig=False, figsize=None, inches_per_pixel=0.1, save=None,
             xticks=None, yticks=None, xlabel=None, ylabel=None, contour=False, sigma=3,
             cmap=None, bound=None, vmin=None, vmax=None, colorbar=False, p=1, l=0.6, s=1.0, alpha_max=1.0):
    """ Plots the image `img` (H..., W..., [C,]). The axes of H and W represent nested subimages.
    `bound` is the optional scale to use for the colormap. Also add an optional title `title` and subdivision lines.
    :param (x|y)ticks: tick labels for the x/y axes at every pixel location
    ;param (x|y)label: legend for the x/y axes.
    """
    img = handle_img(to_numpy(img), p, l, s, alpha_max)
    n = img.ndim // 2
    H_shape, W_shape, C = img.shape[:n], img.shape[n:-1], img.shape[-1]
    H, W = np.prod(H_shape), np.prod(W_shape)
    img = img.reshape((H, W, C))
    
    if not has_fig:
        if figsize is None:
            # TODO change dpi of figure instead
            figsize = inches_per_pixel * W, inches_per_pixel * H
        plt.figure(figsize=figsize)

    if C == 1:
        if cmap is None:
            cmap = "bwr"
        if cmap == "bwr" and bound is None:
            bound = np.abs(img).max()
        if bound is not None:
            vmin = -bound
            vmax = bound
        kwargs = dict(cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        kwargs = dict()
    im = plt.imshow(img, **kwargs, interpolation="nearest")
    if colorbar:
        add_colorbar(im)
        
    if contour:
        img_contour = img[:, :, 0]
        img_contour = gaussian_filter(img_contour, sigma=(sigma, sigma), mode="nearest")
        plt.contour(img_contour, cmap=cmap, vmin=vmin, vmax=vmax)

    # Draw lines to signify sub-images.
    for shape, method in [(H_shape, plt.axhline), (W_shape, plt.axvline)]:
        strides = np.cumprod(shape[::-1])  # number of pixels of sub-images of all depths, starting from smallest to largest
        # Last element of stride is the size of the entire image.
        for i, stride in enumerate(strides[:-1], start=1):
            for pos in np.arange(0, strides[-1] + 1, stride):
                method(pos - 0.5, c="gray", lw=i * 1.5)  # pos is the center of the pixel, -0.5 gives the bottom left corner.
                # 1.5 is the default linewidth, multiply by the nesting level to get larger and larger lines.
        
    format_plot(has_fig=has_fig, save=save, title=title, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel)
        
        
def plot_points(points, has_fig=False, figsize=None, title=None, xlabel=None, ylabel=None, 
                sizes=None, colors=None, markers=None, metric="euclidean"):
    """ Creates a scatter plot of the given points, using UMAP if necessary.
    :points: (N, d) array, uses UMAP if d != 2
    :sizes: float or (N,) array, radii of points in points (default 6)
    :colors: color or (N,) array, colors of points
    :markers: string or (N,) array, markers of points
    :return: the embeddings computed with UMAP
    """
    if not has_fig:
        plt.figure(figsize=figsize)
    
    points = to_numpy(points)
    if points.shape[1] > 2:
        import umap
        reducer = umap.UMAP(metric=metric)
        points = reducer.fit_transform(points)
        xlabel = "UMAP 1"
        ylabel = "UMAP 2"
        
    def to_array(x, default):  # Convert to array if sequence, otherwise repeat single value.
        if x is None:
            return np.full(len(points), default)
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, tuple) or isinstance(x, list):
            return np.array(x)
        else:
            return np.full(len(points), x)

    sizes = to_array(sizes, default=6.) ** 2
    colors = to_array(colors, default="C0")
    markers = to_array(markers, default="o")
    
    # Need to do separate calls for different markers because matplotlib cannot handle it.
    for marker in np.unique(markers):
        I = markers == marker
        plt.scatter(*points[I].T, s=sizes[I], c=colors[I], marker=marker)
        
    plt.gca().set_aspect("equal")
    format_plot(has_fig=has_fig, title=title, xlabel=xlabel, ylabel=ylabel)
        
    return points


def plot_stacked_bar(data, series_labels, title=None):
    """ Plots a stacked bar chart with the data and labels provided.
    :param data: 2-dimensional numpy array or nested list containing data for each series in rows
    param series_labels: list of series labels (these appear in the legend)
    :param title:
    """
    data = to_array(data)
    ny = data.shape[1]
    ind = list(range(ny))
    cum_size = np.zeros(ny)

    plt.figure(figsize=(15, 10))
    plt.title(title)
    for i, row_data in enumerate(data):
        plt.bar(ind, row_data, bottom=cum_size, label=series_labels[i])
        cum_size += row_data
    plt.figlegend(loc="lower center", ncol=10)
    plt.axis("off")
    plt.xlim(-0.5, ny - 0.5)
    plt.ylim(0, 1)
    plt.show()


def create_fig(rows, cols, title=None, s=1.75, wspace=0.01, hspace=0.3, top=None):
    """ Create a figure with associated grdispec and optional title.
    rows is the number of rows, hence, the size of the vertical axis.
    cols is the number of columns, hence, the size of the horizontal axis.
    s is the scale to use (inches per row/col), or size of the figure (horizontal, vertical).
    wspace is the portion of the width reserved for horizontal spacing between subplots.
    hspace is the portion of the height reserved for vertical spacing between subplots.
    """
    if not isinstance(s, tuple):
        s = (cols * s, rows * s)
    fig = plt.figure(facecolor="white", figsize=s)
    if title is not None:
        fig.suptitle(title, fontsize=15)
    #     if top is None:
    #         top = 1 - 0.65 / (rows * s)
    gs = fig.add_gridspec(nrows=rows, ncols=cols, wspace=wspace, hspace=hspace, top=top)
    return fig, gs


def format_plot(has_fig=False, save=None, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None):
    """ Format axes and labels, and show the plot if necessary. """
    if xticks is not None:
        plt.xticks(ticks=np.arange(len(xticks)), labels=xticks, rotation="vertical")
    else:
        plt.xticks(ticks=[], labels=[])
    if yticks is not None:
        plt.yticks(ticks=np.arange(len(yticks)), labels=yticks)
    else:
        plt.yticks(ticks=[], labels=[])
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if save is not None:
        savefig(save)
    if not has_fig:
        plt.show()
        
        
def add_colorbar(im, aspect=20, pad_fraction=1.0, label=None, **kwargs):
    """ Add a vertical color bar to an image plot.
    :param im: output of imshow
    :param aspect: aspect ratio of the colorbar
    :param pad_fraction: padding between the image and the colorbar, as a fraction of the colorbar's width
    :param label: add a label to the colorbar
    :param kwargs: additional kwargs to pass to the colorbar call
    """
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    colorbar = im.axes.figure.colorbar(im, cax=cax, **kwargs)

    if label is not None:
        colorbar.ax.get_yaxis().labelpad = 15
        colorbar.ax.set_ylabel(label, rotation=90)
        
    return colorbar


def arrange_images(images, aspect_ratio=np.sqrt(2)):
    """ (N, H..., W..., [C]) to (h, H..., w, W..., [C]), where w*W/h*H is approximately aspect_ratio. """
    images = to_numpy(images)
    
    n = (images.ndim - 1) // 2
    N, H_shape, W_shape, C_shape = images.shape[0], images.shape[1:n+1], images.shape[n+1:2*n+1], images.shape[2*n+1:]
    # print(f"{N=} {H_shape=} {W_shape=} {C_shape=}")
    
    H, W = np.prod(H_shape), np.prod(W_shape)
    # w/h approx H/W * aspect_ratio, and w*h >= N
    # Leads to w = sqrt(N * H/W * apect_ratio) then rounding up
    w = math.ceil(np.sqrt(N * H / W * aspect_ratio))
    h = math.ceil(N / w)
    # Sometimes w can be reduced without changing h:
    w = math.ceil(N / h)

    res = np.concatenate((images, np.zeros((h * w - N,) + H_shape + W_shape + C_shape)))  # (h * w, H..., W...)
    res = np.moveaxis(res.reshape((h, w) + H_shape + W_shape + C_shape), source=1, destination=n + 1)  # (h, H..., w, W...)
    # print(f"Handled {images.shape=} to {res.shape=}")
    return res


def handle_img(img, p=1, l=0.6, s=1.0, alpha_max=1.0):
    """ Put one or several real/complex images in the right format: (H..., W..., [C,]) to (H..., W..., C) real. """
    if img.dtype in [np.complex64, np.complex128]:
        img = complex_to_rgb(img, p, l, s, alpha_max)
    if img.ndim % 2 == 0:
        img = img[..., None]  # grayscale image
    return img


def complex_to_rgb(x, p, l, s, alpha_max):
    """ Compute the color of the complex coefficients,
    with argument -> hue and modulus -> luminance, at fixed saturation. (*) to (*, 3). """
    from colorsys import hls_to_rgb

    mod = np.abs(x) ** p  # (*)
    alpha = alpha_max * mod / mod.max()  # (*)

    arg = np.angle(x)  # (*)
    h = (arg + np.pi) / (2 * np.pi) + 0.5  # (*)

    c = np.array(np.vectorize(hls_to_rgb)(h, l, s)).transpose(tuple(1 + i for i in range(x.ndim)) + (0,))  # (*, 3)
    c = np.concatenate((c, alpha[..., None]), axis=-1)  # (*, 4)
    return c


# Notebook-specific plotting functions.

def plot_class_sensitivities(exp):
    """ Plot the class sensitivities of all atoms. """
    I = exp.rank_to_index
    plot_img(exp.class_sensitivity[I].T, title=f"Class sensitivities of {exp.exps.short_names[exp.name]}", 
             xlabel="Rank of atom", yticks=exp.exps.class_labels)
    
    
def plot_class_gram_matrix(exp):
    """ Plot the Gram matrix of class-sensitivies to see a correlation-like matrix between classes. """
    plot_img(exp.class_sensitivity.T @ exp.class_sensitivity,
             title=f"Class-Gram matrix of {exp.exps.short_names[exp.name]}", 
             xticks=exp.exps.class_labels, yticks=exp.exps.class_labels)
    
    
def P1_ticks(L):
    return [r"$\phi$"] + sum([[""] * (L//4 - 1) + [rf"$\psi_{l}$"] for l in ["a", "h", "d", "v"]], start=[])
    
    
def plot_pj_weights(exp, j=1, ranks=None, by_rank=True, figsize=None):
    """ Plots the weights of Pj. Only j=1 and L=4 is currently implemented. """
    assert j == 1
    matrix = exp.weight_matrices[j-1]
    
    if ranks is None:
        ranks = np.arange(exp.num_atoms)
    if by_rank:
        matrix = matrix[exp.rank_to_index[ranks]]
        ylabel = "Rank"
    
    else:
        matrix = matrix[ranks]
        ylabel = "Index"
        
    # Set the norm of each atom to 1:
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        
    L = matrix.shape[1] - 1
    xticks = P1_ticks(L=L)
    plot_img(matrix, title=f"Weights of P{j} for {exp.exps.short_names[exp.name]}", xticks=xticks, ylabel=f"{ylabel} of atom",
             figsize=figsize)
    
    
def plot_lucent_atoms(exp, j=1, stationary=True, ranks=None, by_rank=True):
    """ Plots lucent visualizations of atoms. """
    from .lucent_hack import visualize_atom

    if ranks is None:
        ranks = range(exp.num_atoms)
    for i in ranks:
        print(f"Visualizations of atom of {'rank' if by_rank else 'index'} {i} from {exp.exps.short_names[exp.name]}", flush=True)
        visualize_atom(model=exp.model, layer=f"module_{j}_module_3_submodules_dict_(0, 0)",
                       atom=exp.rank_to_index[i] if by_rank else i, stationary=stationary)


def get_atoms(exps, j=1, epoch=None):
    """ Returns atoms (N, D) from single exp or a dictionary of exps (e.g., several random initializations). """
    if not isinstance(exps, dict):
        exps = {"": exps}
    atoms = np.concatenate([exp.weight_matrices(epoch=epoch)[j-1] for exp in exps.values()])  # (N, D)
    # atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    return atoms


def get_sigma_ps(exps, j=1, sigma_x=None, equivariant=False):
    """ Extract covariance matrices from several experiments on the same representation.
    :param exps: dictionary of Experiments (or themselves dictionaries of experiments)
    Example: dict(N100=dict(init1=..., init2=...), N200=dict(init1=..., init2=...))
    :param j: scale at which to extract Ps
    :param sigma_x (D, D), covariance matrix of input of P
    :return: diagonalized sigmas (B [+ 1], D, D) and atoms_dict (exp_name -> (N, D)):
    - equivariant=False, sigma_x=None: sigma_p's
    - equivariant=False, sigma_x!=None: sigma_x, sigma_p's
    - equivariant=False, sigma_x!=None: sigma_p's @ sigma_x
    """
    atoms_dict = {name: get_atoms(exp, j) for name, exp in exps.items()}  # name -> (N, D)
    sigma_ps = np.stack([atoms.T @ atoms / atoms.shape[0] for atoms in atoms_dict.values()])  # (B, D, D)

    sigma = sigma_ps if not equivariant else sigma_ps @ sigma_x  # (B, D, D)
    if not equivariant and sigma_x is not None:
        atoms_dict[r"$\Sigma_x$"] = np.empty(sigma_x.shape)
        sigma = np.concatenate((sigma_x[None], sigma))
    return DecomposedMatrix(sigma, decomposition="eig" if equivariant else "eigh"), atoms_dict


def reshape_and_plot_eigenvectors(eigenvectors, j, args, title="", plot_eigenvectors=True):
    # Reshape eigenvectors when we have previous Ps = id.
    eigen_reshaped = eigenvectors
    
    for i in range(j, 0, -1):
        eigen_reshaped = reshape_atoms(eigen_reshaped, L=args.scat_angles[i - 1])  # (N, N_i-1, 1 + L_i, ..., 1 + L_j)
        if i > 1 and not args.Pr_size[i - 2] == "id":
            break
    # eigen_reshaped is (N, N_i-1, 1 + L_i, ..., 1 + L_j) where i = 1 or P_i-1 != id (N_0 = 1)
    # print(f"Reshaped from {eigenvectors.shape} to {eigen_reshaped.shape} (j = {j}, i = {i})")
    
    # Also visualize in eigenvector basis for previous P
    if i > 1:
        # All previous exps should have exactly the same P_i-1, so we take its covariance matrix.
        # TODO: take a better estimate of the covariance matrix from a P_prev with more atoms?
        # atoms_prev, _, _, eigen_prev = get_atom_eigenvectors({"": dummy_exp}, j=i - 1)  # (N_i-1, D), (N', D)
        # contraction_matrix = eigen_prev @ atoms_prev.T  # (N', N_i-1)
        # eigen_eigen = contract(eigen_reshaped, contraction_matrix, axis=1)  # (N, N', 1 + L_i, ..., 1 + L_j)
        eigen_eigen = None  # Disabled for now
    else:
        eigen_reshaped = eigen_reshaped[:, 0]  # (N, 1 + L_1, ..., 1 + L_j)  # Slightly ugly but simpler in practice
        eigen_eigen = None
        
    title = f"ovariance matrix of atoms with {title}"
#     plot_img(cov, title="C" + title,
#              colorbar=True, figsize=(10,10))
        
    if plot_eigenvectors:
        for eigen, title2 in [(eigen_reshaped, ""), (eigen_eigen, " (contracted)")]:
            # Subsample eigenvectors?            
            
            yticks = None
            if eigen is None:
                continue

            # Plot eigen_reshaped. Arrange axes depending on j.
            if eigen.ndim == 2:  # eigen_reshaped is (N, 1 + L_1)
                eigen_plot = eigen.T  # (1 + L, N)
                yticks = P1_ticks(L=args.scat_angles[j-1])

            elif eigen.ndim == 3:  # eigen_reshaped is (N, N_j-1, 1 + L_j)
                eigen_plot = arrange_images(eigen)
                # Disabled: should be xticks + wrong formula
                # yticks = P1_ticks(L=args.scat_angles[j-1]) * eigen_plot.shape[0]

            elif eigen.ndim == 4:  # eigen_reshaped is (N, N_j-2, 1 + L_j-1, 1 + L_j)
                eigen_plot = arrange_images(eigen[:, None])  # ([H, 1, N_j-2], [W, 1 + L_j-1, 1 + L_j])

            else:
                assert False

            # print(f"Plotting {eigen_plot.shape}")
            plot_img(eigen_plot, title=f"Eigenvectors of c{title}{title2}", yticks=yticks,
                     #xlabel="Rank of eigenvector", figsize=(10,10),
                    )
    
    return eigen_reshaped, eigen_eigen


def plot_sigma_p(sigma_p, args, title="", j=1, plot_eigenvectors=True, plot_eigenvalues=True, ):
    """ Plot eigenvalues and eigenvectors of the given sigma_p, a DecomposedMatrix (D, D).
    Returns reshaped eigenvectors and eigen_eigens. """
    if plot_eigenvalues:
        plot_lines(sigma_p.eigenvalues, title=f"Spectrum of c{title}", xscale="log", yscale="log")
    
    eigen_reshaped, eigen_eigen = reshape_and_plot_eigenvectors(sigma_p.eigenvectors, j=j, args=args, title=title,
                                                                plot_eigenvectors=plot_eigenvectors)
    
    return eigen_reshaped, eigen_eigen


class SigmaXDotProduct(EuclideanDotProduct):
    """ Dot products between sigma_p @ sigma_x's using the given sigma_x. """

    def __init__(self, sigma_x):
        """ sigma_x is (D, D). """
        super().__init__(num_tensor_axes=2)
        self.sigma_x = sigma_x
        self.sigma_x_inv = np.linalg.inv(sigma_x)

    def dot(self, x, y, batch_axes=0):
        """ Optimized, general dot products between sigma_p @ sigma_x's.
        Complexity is O(B(NM+ND+MD)D^2) in time and O(BNM) in additional space.
        :param x: (B..., N..., D, D)
        :param y: (B..., M..., D, D)
        :param batch_axes: number of batch axes (B...,)
        :return: (B..., N..., M...)
        """
        x, y, batch_shape, x_shape, y_shape, data_shape = self.reshapes(x, y, batch_axes=batch_axes, collapse_data=False)
        # x and y are (B, N/M, D, D).

        # Apply sigma_x to reduce to Euclidean dot product.
        x = x @ self.sigma_x_inv  # (B, N, D, D)
        y = self.sigma_x @ y  # (B, M, D, D)

        # Compute Euclidean dot product between x and y.
        dot_products = super().dot(x, y, batch_axes=1)  # (B, N, M)
        dot_products = dot_products.reshape(batch_shape + x_shape + y_shape)  # (B..., N..., M...)
        return dot_products
    
    
def stability_curves(sigma_ps, sigma_x=None, stds=[1/20], ret="full"):
    """ Returns (S, M, B, B) (re="full") or (S, M, B choose 2) (ret="pairs") stability curves.
    S is the number of bin standard deviations, M the number of bin means, and B the number of covariance matrices. """
    means = sigma_ps.eigenvalues.T  # (B, N) to (M, B) with M = N
    stds = np.array([1/20]) # (S,) with S = 1

    # Compute new eigenvalues, of shape (S, M, B, N).
    binned_eigenvalues = sigma_ps.eigenvalues * (np.exp(-(np.log10(sigma_ps.eigenvalues) - np.log10(means[..., None])) ** 2 / (2 * stds[:, None, None, None] ** 2)))
    # NOTE: perhaps we could reduce the memory cost here, by never computing the binned covariance matrices
    # but rather computing the dot products directly.
    binned_sigma_ps = sigma_ps.with_eigenvalues(binned_eigenvalues).matrix  # (S, M, B, D, D)

    if sigma_x is None:
        dot = EuclideanDotProduct(num_tensor_axes=2)
    else:
        dot = SigmaXDotProduct(sigma_x)
    correlations = dot.symmetric_dot(binned_sigma_ps, batch_axes=2, abs=False, ret=ret)  # (S, M, B, B) or (S, M, B choose 2)
    return correlations


def analyze_Ps(exps_dict, j=1, plot=True, plot_eigenvectors=True):
    """ Plot everything there is to know about Ps of same dimension, given a dictionary of title: exps. """

    dummy_exp = list(exps_dict.values())[0]
    args = dummy_exp.args

    sigma_ps, atoms_dict = get_sigma_ps(exps_dict, j=j)  # DecomposedMatrix (B [+ 1], D, D), (name -> (N, D))
    titles = list(exps_dict.keys())

    if plot and plot_eigenvectors:
        for i, title in enumerate(titles):
            # Visualize only eigenvalues and eigenvectors that are not in the kernel.
            eigen_reshaped, eigen_eigen = plot_sigma_p(sigma_ps[i, :atoms_dict[title].shape[0]], args, title=title, j=j,
                                                       plot_eigenvalues=False)

    #             dot_products = eigenvectors @ atoms.T  # (num_eigs, N)
    #             plt.figure()
    #             plt.title(f"Dot products with eigenvectors for {title}")
    #             x = np.linspace(-1, 1, 100)
    #             for i in range(len(dot_products)):
    #                 density = gaussian_kde(dot_products[i])
    #                 plt.plot(x + i*0.05*8/L, density(x) + i*0.2, c="C0")
    #             plt.show()

    # Restrict eigenvectors to maximum number of atoms
    num_atoms = max(atoms_dict[title].shape[0] for title in titles)

    eigen = sigma_ps.eigenvectors
    if plot:
        plot_correlations(eigen[:, :num_atoms], kernel=None,
                          title=f"Correlation between eigenvectors of {', '.join(titles)}")

    # Correlations are (S, M, B choose 2) where B is the number of experiments, M is the number of eigenvalues and
    # S is the number of standard deviations.
    correlations = stability_curves(sigma_ps[:, :num_atoms], ret="pairs")  # (S, M, B choose 2)

    # Plot everything.
    if plot:
        norms = (np.abs(sigma_ps.eigenvectors).sum(-1) - 1) / (np.sqrt(sigma_ps.eigenvectors.shape[-1]) - 1)  # (num_exps, num_vectors)
        stab_curves = correlations[0].T  # (B choose 2, M)

        plot_kwargs = [
#             dict(yscale="log", title=r"Spectra of $\Sigma_P$",
#                  **{title: sigma_ps.eigenvalues[i, :atoms_dict[title].shape[0]] for i, title in enumerate(titles)}),
#             dict(title=r"$\ell^1$-norms of eigenvectors of $\Sigma_P$",
#                  **{title: norms[i, :atoms_dict[title].shape[0]] for i, title in enumerate(titles)}),
            dict(title="Stability curves", ylim=(-0.1,1.1),
                 **{f"{i}": stab_curves[i] for i in range(stab_curves.shape[0])}),
    #                 dict(title="Sqrt-Stability curves", ylim=(-0.1,1.1),
    #                      **{f"{i}": np.sqrt(stab_curves[i]) for i in range(stab_curves.shape[0])}),
        ]


        xscales = ["linear"]
        s = 7.5
        plt.figure(figsize=(len(xscales) * s, len(plot_kwargs) * s))
        i = 0
        for kwargs in plot_kwargs:
            for xscale in xscales:
                i += 1
                plt.subplot(len(plot_kwargs), len(xscales), i)
                plot_lines(xscale=xscale, has_fig=True, **kwargs)
        plt.show()

    return sigma_ps, correlations


def plot_correlations(vectors, title=None, figsize=(10, 10), kernel=None, normalize=True, abs=True, 
                      plot=True, save=None, colorbar=True):
    """ Plot the correlation between sets of vectors.
    :param vectors: (N, M, D) for N experiments, each with M vectors in dimension D
    :param kernel: optional kernel for dot products, can be a dictionary of kernels
    :return: the full correlation matrix (N, M, N, M)
    """
    if kernel is not None:
        assert not normalize  # Normalization not supported for (non-symmetric) kernels.
    if not isinstance(kernel, dict):
        kernel = {(i, j): kernel for i in range(len(vectors)) for j in range(len(vectors))}
    
    matrix = torch.stack([
        torch.stack([EuclideanDotProduct(kernel=kernel[i,j]).dot(vectors[i], vectors[j], normalize=normalize, abs=abs)
                  for j in range(len(vectors))], dim=1) for i in range(len(vectors))], dim=0)  # (N, M, N, M)
    if plot:
        plot_img(matrix, colorbar=colorbar, cmap="viridis", figsize=figsize, vmin=0, vmax=1, 
                 title=title, save=save)
    return matrix


def correlation_matrix(x, y, method="pearson", abs=True, kernel=None):
    """ Returns the correlation matrix between every pair of vectors. TODO: somewhat deprecated, should use DotProduct.
    :param x: 1st set of vectors, of shape (N, D)
    :param y: 2nd set of vectors, of shape (M, D)
    :param method: either "pearson" (cosine similarity) or "spearman" (rank-based)
    :param abs: whether to take the absolute value
    :param kernel: optional (D, D) spd matrix to define the dot product for Pearson correlation
    :return (N, M) correlation matrix, with values in [-1, 1] or [1, 1]
    """
    if method == "pearson":
        corrs = x @ y.T if kernel is None else x @ kernel @ y.T
        norm = lambda a: np.sqrt(np.sum(a ** 2 if kernel is None else a * (a @ kernel), axis=1))
        corrs /= norm(x)[:, None] * norm(y)[None, :]
    else:
        import scipy.stats
        corrs = np.array([[scipy.stats.spearmanr(x[i], y[j])[0] for j in range(y.shape[0])] for i in range(x.shape[0])])
    if abs:
        corrs = np.abs(corrs)
    return corrs


# LaTeX stuff

def latexify(fig_width=None, fig_height=None):
    fig_width_pt = 307.28987   # Get this from LaTeX using \the\textwidth (Here set for Beamer)
    inches_per_pt = 1.0/72.27  # Convert pt to inch

    if fig_width is None:
        fig_width = fig_width_pt*inches_per_pt

    if fig_height is None:
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
      'backend': 'ps',
      #'text.latex.preamble': ['\usepackage{gensymb}'],
#       'axes.labelsize': 8, # fontsize for x and y labels (was 10)
#       'axes.titlesize': 8,
#       'font.size':       8, # was 10
#       'legend.fontsize': 8, # was 10
#       'xtick.labelsize': 8,
#       'ytick.labelsize': 8,
      'text.usetex': True,
      'figure.figsize': [fig_width,fig_height],
      'font.family': 'sans serif'
    }

    matplotlib.rcParams.update(params)
    
def savefig(path):
    """ Saves the current matplotlib figure to the current path.
    Takes care of transparency, padding, resolution and pdf output.
    """
    if not path.endswith(".pdf"):
        path = f"plots/{path}.pdf"
    plt.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0, dpi=600)
