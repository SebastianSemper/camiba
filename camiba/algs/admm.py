import numpy as np
import numpy.linalg as npl
from scipy.linalg import toeplitz as spToep
from scipy.linalg import circulant as spCirc

from ..linalg.basic import soft_thrshld

def ToepAdj(A):
    r"""
    Calc adjoint of a single-level Toeplitz matrix
    """
    n = A.shape[0]
    u = np.empty(n, A.dtype)
    for ii in range(n):
        u[ii] = np.trace(A[:(n-ii), ii:])
    u[0] *= .5
    return u


def BPDN_1D(
    A,              # system matrix
    b,              # measurements
    x_init,         # inital iterate
    rho,            # parameter of augmented lagrangian
    alpha,          # thresholding parameter
    num_maxsteps    # maximum number of steps
):
    """
    Basis Pursuit Denoising
    """
    numN, numM = A.shape

    x = np.zeros(numM)
    z = np.copy(x_init)
    u = np.zeros(numM)

    AAt = A.dot(A.conj().T)

    P = np.eye(numM) - A.conj().T.dot(npl.solve(AAt, A))

    q = A.conj().T.dot(npl.solve(AAt, b))

    x_hat = alpha*x + (1-alpha)*z
    u = u + (x_hat - z)

    for ii in range(num_maxsteps):
        x = P.dot(z-u) + q

        x_hat = alpha*x + (1-alpha)*z
        z = soft_thrshld(x_hat+u, 1.0/rho)

        u = u + (x_hat - z)

    return z


def ANM_LSE_1D(
    A,          # compression matrix
    AH,         # A^H
    AHA,        # A^H * A
    y,          # measurements
    rho,        # augmented lagrangian parameter
    tau,        # regularization parameter
    steps       # maximum number of steps
):
    """
    Atomic Norm Denoising for 1D Line Spectral Estimation


    """
    dtype = np.promote_types(A.dtype, y.dtype)

    K, L = A.shape

    I = np.eye(L)
    e1 = -L * .5 * (tau / rho) * I[0]
    rhoInv = 1. / rho
    tauHalf = -.5 * tau

    Inv = npl.inv(AHA + 2 * rho * I)

    x = np.zeros((L), dtype)
    t = 0
    u = np.zeros((L), dtype)
    T = np.zeros((L + 1, L + 1), dtype)
    Lb = np.zeros((L + 1, L + 1), dtype)
    Z = np.zeros((L + 1, L + 1), dtype)

    Winv = 1. / (np.linspace(L, 1, L))
    Winv[0] = 2. / L

    for ii in range(steps):

        t = Z[L, L] + rhoInv * (Lb[L, L] + tauHalf)

        x = Inv.dot(
            AH.dot(y) + 2 * (Lb[:L, L] + rho * Z[:L, L])
        )

        u = Winv * (
            ToepAdj(Z[:L, :L] + rhoInv * Lb[:L, :L]) + e1
        )

        T[:L, :L] = spToep(u.conj())
        T[:L, L] = x
        T[L, :L] = np.conj(x)
        T[L,  L] = t

        Zspec, Zbase = npl.eigh(T - rhoInv * Lb)
        arrPos = (Zspec > 0)

        Z = Zbase[:, arrPos].dot((Zspec[arrPos] * Zbase[:, arrPos].conj()).T)
        Z = 0.5 * (Z + Z.T.conj())

        Lb += rho * (Z - T)

    return (x, T[:L, :L], t)
