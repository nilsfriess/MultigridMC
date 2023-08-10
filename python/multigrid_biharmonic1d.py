"""Multigrid for the 1d Biharmonic equation

This is a toy implementation which was used to inform the
two-dimensional C++ code
"""

import numpy as np
from matplotlib import pyplot as plt

biharmonic = True


def gauss_seidel(b, x, A, direction="forward"):
    n, _ = A.shape
    if direction == "forward":
        directed_range = range(n)
    else:
        directed_range = range(n - 1, -1, -1)
    for j in directed_range:
        r = b[j] - np.dot(A[j, :], x)
        x[j] += 1 / A[j, j] * r


def prolong(x):
    """linear interpolation"""
    n = x.size + 1
    x_fine = np.zeros(2 * n - 1)
    for j in range(n - 1):
        x_fine[2 * j + 0] += 0.5 * x[j]
        x_fine[2 * j + 1] += x[j]
        x_fine[2 * j + 2] += 0.5 * x[j]
    return x_fine


def restrict(x):
    """restrict with full weighting"""
    n = x.size + 1
    x_coarse = np.zeros(n // 2 - 1)
    for j in range(n // 2 - 1):
        x_coarse[j] = 0.25 * (x[2 * j] + 2 * x[2 * j + 1] + x[2 * j + 2])
    return x_coarse


def prolongation_matrix(n):
    P = np.zeros((n - 1, n // 2 - 1))
    for j in range(n // 2 - 1):
        P[2 * j, j] = 0.5
        P[2 * j + 1, j] = 1.0
        P[2 * j + 2, j] = 0.5
    return P


def discretisation_matrix(n):
    h = 1 / n
    A = np.zeros((n - 1, n - 1))
    if biharmonic:
        for j in range(n - 1):
            A[j, j] = 6.0 / h**4
            if j > 0:
                A[j, j - 1] = -4.0 / h**4
            else:
                A[j, j] += 1.0 / h**4
            if j < n - 2:
                A[j, j + 1] = -4.0 / h**4
            else:
                A[j, j] += 1.0 / h**4
            if j > 1:
                A[j, j - 2] = 1.0 / h**4
            if j < n - 3:
                A[j, j + 2] = 1.0 / h**4
    else:
        for j in range(n - 1):
            A[j, j] = 2.0 / h**2 + 1.0
            if j > 0:
                A[j, j - 1] = -1.0 / h**2
            if j < n - 2:
                A[j, j + 1] = -1.0 / h**2

    return A


def vcycle(b, n_presmooth, n_postsmooth, gamma, A):
    n = b.size + 1
    x = np.zeros(n - 1)
    # A = discretisation_matrix(n)
    if n > 8:
        for k in range(gamma):
            # presmooth
            for j in range(n_presmooth):
                gauss_seidel(b, x, A, direction="forward")
            r = b - A @ x
            P = prolongation_matrix(n)
            b_coarse = P.T @ r
            # b_coarse = restrict(r)
            A_coarse = P.T @ A @ P
            # A_coarse = discretisation_matrix(n // 2)
            x_coarse = vcycle(b_coarse, n_presmooth, n_postsmooth, gamma, A_coarse)
            # x += prolong(x_coarse)
            x += 2 * P @ x_coarse
            # postsmooth
            for j in range(n_postsmooth):
                gauss_seidel(b, x, A, direction="backward")
    else:
        x = np.linalg.inv(A) @ b

    return x


def multigrid_solve(b, n_presmooth=2, n_postsmooth=2, gamma=1):
    n = b.size + 1
    A = discretisation_matrix(n)
    x = np.zeros(n - 1)
    r_norm = np.linalg.norm(b)
    r0_norm = r_norm
    k = 0
    print(f"{k:6d} : {r_norm:8.4e}    {r_norm/r0_norm:8.4e}")
    for k in range(20):
        r = b - A @ x
        rold_norm = r_norm
        r_norm = np.linalg.norm(r)
        if k > 0:
            rho = r_norm / rold_norm
            print(f"{k:6d} : {r_norm:8.4e}    {r_norm/r0_norm:8.4e}    {rho:6.3f}")
        dx = vcycle(r, n_presmooth, n_postsmooth, gamma, A)
        x += dx


n = 256
b = np.random.normal(size=n - 1)
multigrid_solve(b, n_presmooth=2, n_postsmooth=2, gamma=2)
