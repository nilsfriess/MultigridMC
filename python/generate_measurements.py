"""Generate measurements for 2d operator

randomly choose a number of sample points in the unit
square. Construct a random covariance matrix for these
measurements.
"""
import itertools
import numpy as np
from matplotlib import pyplot as plt


def dist_periodic(x, y):
    """Compute distance between two points, taking into account
    periodic boundary conditions"""
    dim = x.shape[0]
    return min(
        [
            np.linalg.norm(x - y + np.asarray(offset))
            for offset in itertools.product([-1.0, 0.0, +1.0], repeat=dim)
        ]
    )


def sample_points(n, dim, dmin=0.1):
    """Draw random points in unit square

    :arg n: number of sample points to draw
    :arg dim: dimension
    :arg dmin: minimum distance between any pair of points
    """
    rng = np.random.default_rng(seed=2154157)
    points = []
    while len(points) < n:
        x = np.asarray(rng.uniform(low=0.0, high=1.0, size=dim))
        if all([dist_periodic(x, p) > dmin for p in points]):
            points.append(x)
    return points


def average(n, mu_low, mu_high):
    """Create array of measured average values

    The measured averages are assumed to be uniformly distributed
    in the interval [mu_low,mu_high].

    :arg n: number of measurement points
    :arg mu_low: lower bound on average
    :arg mu_high: upper bound on average
    """
    rng = np.random.default_rng(seed=2513267)
    return rng.uniform(size=n, low=mu_low, high=mu_high)


def covariance_matrix(n, sigma_low, sigma_high):
    """Create positive definite random covariance matrix

    The eigenvalues of the matrix are constructed such that they are
    in the range [sigma_min,sigma_max]

    :arg n: size of matrix (= number of meaurements)
    :arg sigma_low: lower bound on eigenvalues
    :arg sigma_high: upper bound on eigenvalues
    """
    rng = np.random.default_rng(seed=2511541)
    A_random = rng.uniform(size=(n, n))
    Sigma_diag = np.diag(rng.uniform(low=sigma_low, high=sigma_high, size=n))
    Q, _ = np.linalg.qr(A_random)
    Sigma = Q @ Sigma_diag @ Q.T
    return Sigma


nmeas = 8
dim = 2
dmin = 0.1
p = np.asarray(sample_points(nmeas + 1, dim, dmin))
mean = average(nmeas, 1.0, 4.0)
Sigma = covariance_matrix(nmeas, 1.0, 2.0)

# Print results in a format that can be used in the configuration file
print("n = ", nmeas, ";")
print("measurement_locations = ", repr(list(p[:-1, :].flatten())), ";")
print("sample_location = ", repr(list(p[-1, :].flatten())), ";")
print("mean = ", repr(list(mean.flatten())), ";")
print("covariance = ", repr(list(Sigma.flatten())), ";")

plt.clf()
plt.plot(p[:, 0], p[:, 1], linewidth=0, markersize=4, marker="o")
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.savefig("points.pdf", bbox_inches="tight")
