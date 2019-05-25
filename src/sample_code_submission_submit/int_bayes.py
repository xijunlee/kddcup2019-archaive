import numpy as np
import math
import copy
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform

from sklearn.gaussian_process.kernels import RBF
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor

def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale

# adapted from scipy/optimize/optimize.py for functions with 2d output
def _approx_fprime(xk, f, epsilon, args=()):
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk), ), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad


class IntKernel(RBF):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5, int_dim=None):
        RBF.__init__(self, length_scale, length_scale_bounds)
        self.nu = nu
        self.int_dim = int_dim

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = copy.deepcopy(X)
        if self.int_dim is not None:
            if not isinstance(self.int_dim, list):
                raise TypeError('int_dim should be a list of dimensions of integer features.')

            for i in self.int_dim:
                if not isinstance(i, int):
                    raise TypeError('The element of list int_dim should be integers.')
                # X[:, i] = np.apply_along_axis(int, 1, X[:, i])
                X[:, i] = X[:, i].astype(np.int)

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric='euclidean')
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='euclidean')

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1. + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = (math.sqrt(2 * self.nu) * K)
            K.fill((2 ** (1. - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
            else:
                D = squareform(dists ** 2)[:, :, np.newaxis]

            if self.nu == 0.5:
                K_gradient = K[..., np.newaxis] * D \
                             / np.sqrt(D.sum(2))[:, :, np.newaxis]
                K_gradient[~np.isfinite(K_gradient)] = 0
            elif self.nu == 1.5:
                K_gradient = \
                    3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu)
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0],
                self.nu)


class BayesianOptimizationIntKernel(BayesianOptimization):
    def __init__(self, f, pbounds, int_dim=None, random_state=None, verbose=2):
        BayesianOptimization.__init__(self, f, pbounds, random_state, verbose)

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=IntKernel(int_dim=int_dim, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=self._random_state,
        )
