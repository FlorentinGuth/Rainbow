""" Handles linear-algebra utilities. """

import numpy as np
import torch


default_backend = torch


def get_backend(x):
    """ Returns the backend adapted to a given tensor or array. """
    if isinstance(x, torch.Tensor):
        return torch
    else:
        return np


def relative_error(target, estimation):
    """ Computes the relative error between a target and an estimation. """
    def norm(x):
        backend = get_backend(x)
        return backend.linalg.norm(x.flatten())
    return norm(target - estimation) / norm(target)


def tensor_summary_stats(x):
    """ Returns summary statistics about x: shape, mean, std, min, max. """
    return f"shape {x.shape}, values in [{x.min():.3f}, {x.max():.3f}] and around {x.mean():.3f} +- {x.std():.3f}"


class DotProduct:
    """ Class which handles logic behind dot products. A subclass needs only implementing dot.
    Current small inefficiencies: could do in-place operations.
    """

    def __init__(self, num_tensor_axes=1):
        self.num_tensor_axes = num_tensor_axes

    def contract_shapes(self, *arrays, batch_axes=0):
        """ Utility function to reshape arrays into more convenient forms. Inverse of expand_shapes in some sort.
        :param arrays: i-th array is of shape (B..., Ni..., Di...)
        :param batch_axes: number of batch axes (B...,)
        :return: tuple of (B, Ni, Di...), batch_shape, and tuple of (Ni...)-shapes
        """
        batch_shape = arrays[0].shape[:batch_axes]
        assert all(batch_shape == array.shape[:batch_axes] for array in arrays[1:])
        b = np.prod(batch_shape, dtype=int)

        array_shape = tuple(array.shape[batch_axes:-self.num_tensor_axes] for array in arrays)

        def reshape(x):  # (B..., Ni..., D...) to (B, Ni, D[...])
            return x.reshape((b, -1) + x.shape[-self.num_tensor_axes:])

        return *(reshape(array) for array in arrays), batch_shape, *array_shape
    
    def _dot(self, x, y):
        """ Optimized, general dot products between tensors. This is the version that should be overriden in subclasses.
        :param x: (B, N, D...)
        :param y: (B, M, D...)
        :return: (B, N, M)
        """
        raise NotImplementedError

    def dot(self, x, y, batch_axes=0, normalize=False, abs=False):
        """ Optimized, general dot products between tensors.
        Generally complexity is O(BNMD) in time and O(BNM) in additional space.
        :param x: (B..., N..., D...)
        :param y: (B..., M..., D...)
        :param batch_axes: number of batch axes (B...,)
        :param normalize: whether to normalize the tensors
        :param abs: whether to take the absolute value of the dot products
        :return: (B..., N..., M...), with values smaller than 1 in magnitude (if normalize) and non-negative (if abs)
        """
        backend = get_backend(x)
        x, y, batch_shape, x_shape, y_shape = self.contract_shapes(x, y, batch_axes=batch_axes)  # (B, N/M, D...)

        # Normalization of x and y before costs O(B(N+M)D) divisions, and after is O(BNM).
        n, m, d = np.prod(x_shape, dtype=int), np.prod(y_shape, dtype=int), np.prod(x.shape[2:], dtype=int)
        cost_before = (n + m) * d
        cost_after = n * m
        # Disable normalize before for non-symmetric kernels.
        normalize_before = normalize and cost_before < cost_after

        if normalize_before:
            x = self.normalize(x)
            y = self.normalize(y)

        dot_products = self._dot(x, y)  # (B, N, M)

        if normalize and not normalize_before:
            dot_products /= self.norm(x)[:, :, None] * self.norm(y)[:, None, :]  # (B, N, M)
        if abs:
            dot_products = backend.abs(dot_products)

        return dot_products.reshape(batch_shape + x_shape + y_shape)

    def symmetric_dot(self, x, batch_axes=0, normalize=False, abs=False, ret="full"):
        """ Optimized version of dot(x, x).
        ret describes the returned object: can be "full" (B..., N..., N...) or "pairs" (B..., N(N +/- 1)/2) (upper-
        triangular part, with the diagonal only if normalize=False).
        """
        backend = get_backend(x)
        x, batch_shape, x_shape = self.contract_shapes(x, batch_axes=batch_axes)  # (B, N, D...)

        # Normalization of x before costs O(BND) divisions, and after is O(BN²).
        n, d = np.prod(x_shape, dtype=int), np.prod(x.shape[2:], dtype=int)
        cost_before = d
        cost_after = (n - 1) / 2
        normalize_before = normalize and cost_before < cost_after

        if normalize_before:
            x = self.normalize(x)
            
        kwargs = dict(device=x.device) if backend == torch else {}
        def triu_indices(n, k):  # Pytorch has different argument conventions than numpy...
            if backend == torch:
                kw = dict(row=n, col=n, offset=k)
            else:
                kw = dict(n=n, m=n, k=k)
            return backend.triu_indices(**kw, **kwargs)

        # Extract upper-triangular for correlations.
        # If we have normalized, we don't need to compute the diagonal because it is 1.
        # Otherwise, we need to compute it.
        i1, i2 = triu_indices(n, k=1 if normalize else 0)  # pair of (M,) arrays, with M = N(N - 1)/2
        dot_products = self.dot(x[:, i1], x[:, i2], batch_axes=2, abs=abs)  # (B, M)

        if normalize and not normalize_before:
            norms = self.norm(x)  # (B, N)
            dot_products /= norms[:, i1] * norms[:, i2]  # (B, M)

        if ret == "pairs":
            dot_products = dot_products.reshape(batch_shape + (-1,))  # (B..., M)
            return dot_products
        elif ret == "full":
            # Convert to symmetric matrix.
            full_dot_products = backend.empty((dot_products.shape[0], n, n), **kwargs)  # (B, N, N)
            full_dot_products[:, i1, i2] = dot_products  # (B, M)
            if normalize:
                # Fill strictly lower-triangular part, diagonal is only ones.
                full_dot_products[:, i2, i1] = dot_products
                i = backend.arange(n, **kwargs)
                full_dot_products[:, i, i] = 1
            else:
                # Fill strictly lower-triangular part.
                i1, i2 = triu_indices(n, k=1)  # k was 0 before
                full_dot_products[:, i2, i1] = full_dot_products[:, i1, i2]

            full_dot_products = full_dot_products.reshape(batch_shape + x_shape + x_shape)  # (B..., N..., N...)
            return full_dot_products
        else:
            raise ValueError(f"Unknown return value {ret=}")
            
    def square_norm(self, x):
        """ (B..., D...) to (B...,). """
        x, batch_shape, _ = self.contract_shapes(x, batch_axes=x.ndim - self.num_tensor_axes)  # (B, 1, D...)
        norms = self._dot(x, x)  # (B, 1, 1)
        return norms.reshape(batch_shape)
            
    def norm(self, x):
        """ (B..., D...) to (B...,). """
        backend = get_backend(x)
        return backend.sqrt(self.square_norm(x))  # (B...,)

    def normalize(self, x):
        """ (N..., D...) to (N..., D...). """
        return x / self.norm(x)[(...,) + (None,) * self.num_tensor_axes]
                
    def square_dist(self, x, y, batch_axes=0):
        """ Optimized, general square distances between tensors.
        Generally complexity is O(BNMD) in time and O(BNM) in additional space.
        :param x: (B..., N..., D...)
        :param y: (B..., M..., D...)
        :param batch_axes: number of batch axes (B...,)
        :return: (B..., N..., M...)
        """
        num_x_axes = x.ndim - batch_axes - self.num_tensor_axes
        num_y_axes = y.ndim - batch_axes - self.num_tensor_axes
        
        # Expands ||x - y||² = ||x||² + ||y||² - 2<x,y>
        x_square_norm = self.square_norm(x)[(...,) + (slice(None),) * num_x_axes + (None,) * num_y_axes]  # (B..., N..., 1...)
        y_square_norm = self.square_norm(y)[(...,) + (None,) * num_x_axes + (slice(None),) * num_y_axes]  # (B..., 1..., M...)
        x_y_dot = self.dot(x, y)  # (B..., N..., M...)
        
        return x_square_norm + y_square_norm - 2 * x_y_dot


class EuclideanDotProduct(DotProduct):
    """ Euclidean dot product between tensors of a given rank (default 1, i.e., vectors). """

    def __init__(self, num_tensor_axes=1, kernel=None):
        """ If given, kernel is (D, D'). """
        super().__init__(num_tensor_axes=num_tensor_axes)
        self.kernel = kernel

    def _dot(self, x, y):
        """ Optimized, general dot products between tensors.
        Complexity is O(BNMD) in time and O(BNM) in additional space.
        With a kernel, this becomes O(BN(M + D)D) in time, i.e. there is an additional cost O(BND²), assuming N <= M.
        :param x: (B, N, D...)
        :param y: (B, M, D'...)
        :return: (B, N, M)
        """
        x, y = x.reshape(x.shape[:2] + (-1,)), y.reshape(y.shape[:2] + (-1,))
        
        # If necessary, apply kernel to reduce to Euclidean dot product.
        if self.kernel is not None:
            # Apply the kernel on the smallest of the two sets of vectors.
            if x.shape[1] <= y.shape[1]:
                x = x @ self.kernel  # (B, N, D'), computed in time O(BND²)
            else:
                y = y @ self.kernel.T  # (B, M, D), computed in time O(BMD²)

        # Compute Euclidean dot product between x and y.
        dot_products = x @ transpose(y)  # (B, N, M)
        return dot_products
    

dot = EuclideanDotProduct()  # Standard dot product between vectors.


def sparsity(x, axis=-1):  
    """ (*, D) to (*,), 0 is Gaussian and 1 is Dirac (when d -> infinity). Assumes x is normalized. """
    backend = get_backend(x)
    return 1 - backend.abs(x).sum(axis) / np.sqrt(2 * x.shape[-1] / np.pi)


def contract(tensor, matrix, axis):
    """ tensor is (..., D, ...), matrix is (P, D), returns (..., P, ...). """
    backend = get_backend(tensor)
    t = backend.moveaxis(tensor, source=axis, destination=-1)  # (..., D)
    r = t @ matrix.T  # (..., P)
    return backend.moveaxis(r, source=-1, destination=axis)  # (..., P, ...)


def matmul(x, y, batch_axes=0, contracting_axes=1):
    """ Computes a batched matrix multiplication. This is a transposed dot, but eschews the transpose.
    :param x: (B..., N..., D...)
    :param y: (B..., D..., M...)
    :param batch_axes: number of batch axes (B...,)
    :param contracting_axes: number of contracting axes (D...,)
    :return: (B..., N..., M...)
    """
    d = np.prod(x.shape[-contracting_axes])
    
    x_shape = x.shape[batch_axes:-contracting_axes]  # (N...,)
    x = x.reshape(x.shape[:batch_axes] + (-1, d))  # (B..., N, D)
    
    y_shape = y.shape[batch_axes + contracting_axes:]  # (M...,)
    y = y.reshape(y.shape[:batch_axes] + (d, -1))  # (B...., D, M)
    
    res = x @ y  # (B..., N, M)
    res = res.reshape(res.shape[:batch_axes] + x_shape + y_shape)  # (B..., N..., M...)
    return res


def transpose(matrix):
    """ Transpose a stack of matrices: (*, N, M) to (*, M, N). """
    backend = get_backend(matrix)
    return backend.swapaxes(matrix, -1, -2)


def decompose(matrix, decomposition="eigh", rank=None):
    """ Performs the decomposition matrix = eigenvectors.T @ diag(eigenvalues) @ dual_eigenvectors.
    matrix is (*, C, D), returns eigenvalues in descending order (*, N), eigenvectors (*, N, C) and their duals (*, N, D).
    N can be smaller than D because we could prune small eigenvalues.
    There are three decompositions:
    - "svd": compute singular values (non-negative) and left and right singular vectors (orthogonal bases)
    - "eig": compute eigenvalues and eigenvectors, dual eigenvectors are the inverse transpose of eigenvectors (requires C = D and real eigenvalues)
    - "eigh": Hermitian case, equivalent to both "svd" and "eig" but faster and more stable (requires C = D)
    rank is an optional upper bound used to prune the number of eigenvalues and eigenvectors.
    """
    def _decompose(matrix):
        backend = get_backend(matrix)
        if decomposition == "svd":
            eigenvectors, eigenvalues, dual_eigenvectors = backend.linalg.svd(matrix, full_matrices=False)
            # Shapes are (*, N), (*, C, N), (*, N, D) with N = min(C, D). Singular values are in descending order.
            eigenvectors = transpose(eigenvectors)  # (*, N, C)
        else:
            sym = dict(eig=False, eigh=True)[decomposition]
            method = backend.linalg.eigh if sym else backend.linalg.eig
            eigenvalues, eigenvectors = method(matrix)  # eigenvalues (*, N) ascending, eigenvectors (*, C, N)

            # Sort in descending order.
            if sym:
                if backend == np:
                    eigenvalues, eigenvectors = eigenvalues[..., ::-1], eigenvectors[..., ::-1]
                else:
                    eigenvalues, eigenvectors = eigenvalues.flip(-1), eigenvectors.flip(-1)
            else:
                assert np.isreal(eigenvalues.dtype)  # Complex eigenvalues not dealt with for now.
                I = backend.argsort(-eigenvalues, axis=-1)  # (*, N)
                take_along = torch.take_along_dim if backend == torch else np.take_along_axis
                eigenvalues, eigenvectors = take_along(eigenvalues, I, axis=-1), \
                                            take_along(eigenvectors, I[..., None, :], axis=-1)

            eigenvectors = transpose(eigenvectors)  # (*, N, C)

            if sym:
                dual_eigenvectors = eigenvectors
            else:
                dual_eigenvectors = transpose(backend.linalg.inv(eigenvectors))  # (*, N, D)

        # Prune eigenvalues that are theoretically zero because of low-rank.
        if rank is not None:
            eigenvalues = eigenvalues[..., :rank]
            eigenvectors = eigenvectors[..., :rank, :]
            dual_eigenvectors = dual_eigenvectors[..., :rank, :]

        # Prune eigenvalues that are too small (deprecated because dual_eigenvectors and batch axes)
        # I = eigenvalues >= eigenvalues[0] / 1e6
        # eigenvalues, eigenvectors = eigenvalues[I], eigenvectors[I]

        return eigenvalues, eigenvectors, dual_eigenvectors
    
    try:
        eig = _decompose(matrix)
    except RuntimeError as ex:
        print(f"RuntimeError ({ex}) while computing {decomposition} decomposition of shape {matrix.shape}, retrying with numpy")
        matrix_np = matrix.cpu().numpy()
        eigs_np = _decompose(matrix_np)
        eig = tuple(torch.from_numpy(e_np).to(dtype=matrix.dtype, device=matrix.device) for e_np in eigs_np)
        
    return eig


def reconstruct(eigenvalues, eigenvectors, dual_eigenvectors):
    """ eigenvalues is (*, N,), eigenvectors are (*, N, C) and duals are (*, N, D). Returns (*, C, D) matrices.
    Assumes real eigenvalues and eigenvectors. """
    # Reconstruct with eigenvectors.T @ diag(eigenvalues) @ dual_eigenvectors.
    return transpose(eigenvectors) @ (eigenvalues[..., :, None] * dual_eigenvectors)


class DecomposedMatrix:
    """ Class for holding decomposed matrices (diagonalization or SVD). Only real arrays are supported.
    Typical usage: DecomposedMatrix(m).apply(some function).matrix.
    """

    def __init__(self, matrix=None, decomposition="eigh", rank=None,
                 eigenvalues=None, eigenvectors=None, dual_eigenvectors=None):
        """ One typically either gives matrix or eigenvalues, eigenvectors and dual_eigenvectors.
        :param matrix: (*, C, D)
        :param decomposition: any of "eig", "eigh", or "svd"
        :param rank: optional upper bound on the rank to limit the number of eigenvalues and eigenvectors
        :param eigenvalues: (*, R) where R is the rank
        :param eigenvectors: (*, R, C)
        :param dual_eigenvectors (*, R, D)
        """
        self._matrix = matrix  # (*, C, D)
        self.decomposition = decomposition
        self.rank = rank if rank is not None else (min(*matrix.shape[-2:]) if matrix is not None else eigenvectors.shape[-2])
        self._eigenvalues = eigenvalues  # (*, R), descending
        self._eigenvectors = eigenvectors  # (*, R, C)
        self._dual_eigenvectors = eigenvectors if dual_eigenvectors is None else dual_eigenvectors  # (*, R, D)
        
    @property
    def backend(self):
        return get_backend(self._matrix if self._matrix is not None else self._eigenvectors)
        
    @property
    def matrix(self):
        return self.reconstruct()._matrix

    @property
    def eigenvalues(self):
        """ (*, N) """
        return self.decompose()._eigenvalues

    @property
    def eigenvectors(self):
        """ (*, N, C) """
        return self.decompose()._eigenvectors

    @property
    def dual_eigenvectors(self):
        """ (*, N, D) """
        return self.decompose()._dual_eigenvectors
    
    @property
    def singular_values(self):
        """ (*, N), alias for eigenvalues. """
        return self.eigenvalues
    
    @property
    def left_singular_vectors(self):
        """ (*, N, C), alias for eigenvectors. """
        return self.eigenvectors
    
    @property
    def right_singular_vectors(self):
        """ (*, N, D), alias for dual_eigenvectors. """
        return self.dual_eigenvectors

    def decompose(self, do=True) -> "DecomposedMatrix":
        """ Diagonalizes the matrix (if not None and not cached). do=False is a no-op provided for convenience. """
        if do and self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors, self._dual_eigenvectors = decompose(self._matrix, decomposition=self.decomposition, rank=self.rank)
            # Sets the true rank (min(rank, C, D)) as opposed to the optional upper bound provided.
            self.rank = self._eigenvalues.shape[-1]
        return self

    def reconstruct(self, do=True) -> "DecomposedMatrix":
        """ Reconstructs the matrix (if not None and not cached). do=False is a no-op provided for convenience. """
        if do and self._matrix is None:
            self._matrix = reconstruct(self._eigenvalues, self._eigenvectors, self._dual_eigenvectors)
        return self

    def with_eigenvalues(self, eigenvalues) -> "DecomposedMatrix":
        """ Replaces the eigenvalues with a new set (*, N). * can be different as long as it broadcasts. """
        return DecomposedMatrix(
            decomposition=self.decomposition, rank=self.rank,
            eigenvalues=eigenvalues, eigenvectors=self.eigenvectors, dual_eigenvectors=self.dual_eigenvectors,
        )

    def apply(self, f) -> "DecomposedMatrix":
        """ Applies a function on the matrix via its eigenvalues.
        :param f: a scalar function (which can depend on the matrix) : (*, N) -> (*, N)
        """
        return self.with_eigenvalues(eigenvalues=f(self.eigenvalues))

    def sqrt(self) -> "DecomposedMatrix":
        """ Computes the symmetric matrix square root. """
        return self.apply(lambda t: self.backend.sqrt(self.backend.abs(t)))

    def rsqrt(self) -> "DecomposedMatrix":
        """ Computes the reciprocal of the symmetric matrix square root. """
        return self.apply(lambda t: 1 / self.backend.sqrt(t))
    
    def orthogonalize(self) -> "DecomposedMatrix":
        """ Computes the projection to orthogonal matrices by setting all eigenvalues to 1. """
        return self.apply(self.backend.ones_like)
    
    def project(self, rank) -> "DecomposedMatrix":
        """ Lowers the rank of the matrix. rank=None is a no-op. """
        if rank is None or (self.rank is not None and self.rank <= rank):
            return self
        else:
            self.decompose()  # We need to decompose the matrix
            return DecomposedMatrix(
                decomposition=self.decomposition, rank=rank if self.rank is None else min(rank, self.rank),
                eigenvalues=self.eigenvalues[..., :rank],  eigenvectors=self.eigenvectors[..., :rank, :], 
                dual_eigenvectors=self.dual_eigenvectors[..., :rank, :],
            ) 

    def __getitem__(self, item) -> "DecomposedMatrix":
        """ Indexes simultaneously matrices, eigenvalues and eigenvectors to select a subset along the batch axis. """
        def index(m):
            return m[item] if m is not None else None
        return DecomposedMatrix(
            matrix=index(self._matrix), decomposition=self.decomposition, rank=self.rank,
            eigenvalues=index(self._eigenvalues), eigenvectors=index(self._eigenvectors),
            dual_eigenvectors=index(self._dual_eigenvectors),
        )
    
    @property
    def T(self) -> "DecomposedMatrix":
        """ Returns a transposed view of this DecomposedMatrix (swaps eigenvectors and dual_eigenvectors). """
        return DecomposedMatrix(
            matrix=self._matrix.T if self._matrix is not None else None, decomposition=self.decomposition, rank=self.rank,
            eigenvalues=self._eigenvalues, eigenvectors=self._dual_eigenvectors, dual_eigenvectors=self._eigenvectors,
        )
    
    def to(self, dtype=None, device=None) -> "DecomposedMatrix":
        """ Casts the DecomposedMatrix into a different dtype/device. """
        def t(m):
            return m.to(dtype=dtype, device=device) if m is not None else None
        return DecomposedMatrix(
            matrix=t(self._matrix), decomposition=self.decomposition, rank=self.rank,
            eigenvalues=t(self._eigenvalues), eigenvectors=t(self._eigenvectors),
            dual_eigenvectors=t(self._dual_eigenvectors),
        )
    
    @property
    def dtype(self):
        return self._matrix.dtype if self._matrix is not None else self._eigenvectors.dtype
    
    @property
    def device(self):
        return self._matrix.device if self._matrix is not None else self._eigenvectors.device
    
    
def orthogonalize(matrix):
    """ Returns an orthogonalized version of (*, C, D) matrix with the same shape. """
    return DecomposedMatrix(matrix, decomposition="svd").orthogonalize().matrix


def empirical_covariance(x, rank=float("inf")) -> DecomposedMatrix:
    """ Compute covariance along second-to-last axis: (*, N, D) to (*, D, D). rank is an optional upper-bound on the rank.
    Efficient optimization: performs an SVD of x if N < D.
    """
    backend = get_backend(x)
    assert backend == torch
    
    n, d = x.shape[-2:]
    rank = min(n, d, rank)
    
    if n < d:
        # SVD more efficient: we do not compute the large (D, D) matrix.
        x = DecomposedMatrix(x, decomposition="svd", rank=rank)
        # x = U S V^T -> C = X^T X / N = V S²/N V^T
        return DecomposedMatrix(eigenvalues=x.eigenvalues ** 2 / n, eigenvectors=x.dual_eigenvectors, 
                                dual_eigenvectors=x.dual_eigenvectors,
                                decomposition="eigh", rank=rank)
    else:
        # Computing the covariance matrix is more efficient.
        return DecomposedMatrix(x.mT @ x / n, decomposition="eigh", 
                                rank=n if rank is None else min(x.shape[-2], rank))


def random_gaussian(shape, dtype=None, device=None, cov=None, backend=torch):
    """ Returns Gaussian samples of the given covariance.
    :param shape: (*, D)
    :param cov: optional, (*, D, D) DecomposedMatrix or tensor (should be full-rank or suitably pruned)
    :return: Gaussian samples of given shape
    """
    
    if cov is not None:
        if not isinstance(cov, DecomposedMatrix):
            cov = DecomposedMatrix(cov, decomposition="eigh")
        backend = cov.backend
    
    if backend == torch:
        if cov is not None:
            dtype = cov.matrix.dtype
            device = cov.matrix.device
        x = torch.randn(shape, dtype=dtype, device=device)
    else:
        x = np.random.standard_normal(shape)
    
    if cov is not None:
        x = x @ cov.sqrt().matrix
        
    return x


def random_gaussian_like(x):
    """ Returns Gaussian samples of the same shape and covariance as x (*, D) (covariance is (D, D)). """
    return random_gaussian(shape=x.shape, cov=empirical_covariance(x))


def random_orthogonal(shape, dtype=None, device=None, backend=torch):
    """ Returns random orthogonal matrices of the given shape (*, C, D). """
    return orthogonalize(random_gaussian(shape, dtype=dtype, device=device, backend=backend))


def random_orthogonal_like(x):
    """ Returns random orthogonal matrices of the same shape as x (*, C, D). """
    return random_orthogonal(shape=x.shape, dtype=x.dtype, device=x.device)


def marchenko_pastur_eigenvalues(shape, dtype=None, device=None, backend=torch):
    """ Returns a realization of MP eigenvalues for a given shape. Returns descending eigenvalues as a (min(N, D),) tensor. """
    return empirical_covariance(random_gaussian(shape, dtype=dtype, device=device, backend=backend)).eigenvalues


def renormalize_cosine_sim(u, d, negate=False):
    """ Compute element-wise renormalized cosine similarity, ufunc from [0, 1] to [0, 1].
    :param u: numpy array, cosine similarity (absolute value of normalized dot product, in [0, 1])
    :param d: dimension of dot product
    :param negate: whether to compute a similarity (increasing function) 
    or the portion of surface subtended by the angle (decreasing function, one minus the above)
    :return: numpy array, probability of observing a lower (if negate is False) or higher (if negate is True) similarity
    with two random independent vectors on the sphere in dimension d
    """
    from scipy.special import gamma, hyp2f1  # Let's avoid the dependence on scipy with a lazy import

    h = hyp2f1(1/2, (3 - d)/2, 3/2, u ** 2)

    def factor(d):
        """ Compute Gamma(d / 2) / (sqrt(pi) * Gamma((d - 1) / 2)) in a numerically-stable way. """
        assert d % 2 == 0  # Odd d will require a different formula?
        k = d // 2
        i = np.arange(1, 2 * (k - 1))
        return 2 * np.prod(2 * ((i + 1) // 2) / i) / np.pi  # Suspicious factor of 2?
    
    f = factor(d)
    # f = 1 / hyp2f1(1/2, (3 - d)/2, 3/2, 1)  # Numerically unstable (can be NaN for high dimensions)
    # f = gamma(d/2) / (np.sqrt(np.pi) * gamma((d - 1)/2))  # Numerically unstable (overflow for high dimensions)
    
    # print(f"f {f} h {tensor_summary_stats(h)}")
    y = f * u * h
    if negate:
        y = 1 - y
        
    return y


def fit_affine(y, x=None, idx=slice(None), x_log=False, y_log=False):
    """ Fit an affine function.
    :param y: y data
    :param x: x data (optional)
    :param idx: restrict the fit to a range
    :param x_log: whether to apply a log on the x data
    :param y_log: whether to apply a log on the y data
    :return: fitted y values applied on the whole of x, slope, intercept
    """
    backend = get_backend(y)
    
    def transform_x(x):
        """ (N,) to (N, 2). """
        if x_log:
            x = backend.log(x)
        x = backend.stack((x, backend.ones_like(x)), -1)  # (N, 2)
        return x
    
    def transform_y(y):
        """ (N,) to (N,). """
        if y_log:
            y = backend.log(y)
        return y
    
    def transform_inv_y(y):
        """ (N,) to (N,). """
        if y_log:
            y = backend.exp(y)
        return y
    
    if x is None:
        kwargs = dict(dtype=y.dtype, device=y.device) if backend == torch else {}
        x = 1 + backend.arange(len(y), **kwargs)
    x_fit = transform_x(x[idx])
    y_fit = transform_y(y[idx])
    
    # Exclude NaNs (when y has negative values and y_log is True)
    I = ~backend.isnan(y_fit)
    x_fit = x_fit[I]
    y_fit = y_fit[I]
    
    cov_xx = x_fit.T @ x_fit  # (2, 2)
    cov_yx = x_fit.T @ y_fit   # (2,)
    w = backend.linalg.inv(cov_xx) @ cov_yx  # (2,)
    
    def predict(x):
        """ (N,) to (N,). """
        return transform_inv_y(transform_x(x) @ w)
    
    w_np = w.cpu().numpy() if backend == torch else w
    return predict(x), w_np[0], w_np[1]


if __name__ == "__main__":
    def test_correlations():
        for b, n, d in ((10, 100, 5), (10, 5, 100)):
            kernel = np.eye(d)
            for dot in (EuclideanDotProduct(),
                        EuclideanDotProduct(kernel=kernel)):
                for normalize in [False, True]:
                    x = np.random.normal(size=(b, n, d))

                    target = np.sum(x[:, :, None, :] * x[:, None, :, :], axis=-1)  # (b, n, n)
                    if normalize:
                        norms = np.sqrt(np.sum(x ** 2, axis=-1))  # (b, n)
                        target /= (norms[:, None, :] * norms[:, :, None])  # (b, n, n)

                    print(relative_error(target, dot.dot(x, x, batch_axes=1, normalize=normalize, abs=False)))
                    print(relative_error(target, dot.symmetric_dot(x, batch_axes=1, normalize=normalize, abs=False)))

#     test_correlations()

    def test_reconstruct():
        def random_symmetric_matrix(n, d):
            """ Draws a random symmetric matrix of shape (n, d, d) as a covariance matrix of m=2*d normal random vectors. """
            m = 2 * d
            random_vectors = np.random.normal(size=(n, m, d))
            return transpose(random_vectors) @ random_vectors  # (n, d, d)

        n = 100
        d = 20
        for decomposition in ["eig", "svd"]:
            # The product of two symmetric matrices is diagonalizable over R, but not in an orthogonal basis.
            x = random_symmetric_matrix(n, d) @ random_symmetric_matrix(n, d)  # (n, d, d)
            m = DecomposedMatrix(x, decomposition=decomposition)
            print(relative_error(x, m.with_eigenvalues(m.eigenvalues).matrix))

#     test_reconstruct()

    def test_contract():
        b = 10
        c = 20
        d = 10
        m = 5
        n = 5
        for _ in range(3):
            x = random_gaussian((b, c, m, n))
            a = random_gaussian((d, c))
            target = torch.nn.functional.conv2d(x, a[:, :, None, None])
            approx = contract(x, a, axis=1)
            print(relative_error(target, approx))
            
    test_contract()
