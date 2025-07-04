from itertools import batched

import numpy as np
import scipy as sp
from tqdm import tqdm

from .utils import KernelWrapper, oasis, piv_cholesky


def dmd(data, reg: float = 1e-16, rank: int | None = None):
    if rank:
        rank = int(rank)
    reg = float(reg)

    _X, _Y = data[:-1].T, data[1:].T

    U, S, V = sp.linalg.svd(_X, full_matrices=False, check_finite=False)
    cutoff = np.argwhere(S >= S[0] * reg)[-1, 0]
    U, S, V = U[:, :cutoff], S[:cutoff], V[:cutoff]

    rayleigh = ((U.conj().T @ _Y) @ V.conj().T) / S
    evals, tempvecs = sp.linalg.eig(rayleigh, check_finite=False)
    modes = U @ tempvecs

    res = np.linalg.norm(((_Y @ V.conj().T) / S) @ tempvecs - modes * evals, axis=0)
    modes = modes.T
    perm = res.argsort()
    evals, modes, res = evals[perm], modes[perm], res[perm]
    if rank is not None:
        _rank = min(cutoff, rank)
        return evals[:_rank], modes[:_rank], res[:_rank]
    return evals, None, modes, res


def kdmd(
    data: np.ndarray,
    kernel: KernelWrapper,
    return_modes: bool = False,
    reg: float = 1e-16,
    rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return_modes, reg = bool(return_modes), float(reg)
    if rank:
        rank = int(rank)

    kernel_mat = kernel(data)
    G, A = kernel_mat[:-1, :-1], kernel_mat[1:, :-1]

    tempvals, tempvecs = sp.linalg.eigh(G, check_finite=False)
    perm = tempvals.argsort()[::-1]
    tempvals, tempvecs = tempvals[perm], tempvecs[:, perm]
    cutoff = np.argwhere(tempvals >= tempvals[0] * reg)[-1, 0]
    tempvals, tempvecs = tempvals[:cutoff], tempvecs[:, :cutoff]
    scal_tempvecs = tempvecs / np.sqrt(tempvals)

    K_approx = scal_tempvecs.T @ A @ scal_tempvecs
    evals, lvecs, rvecs = sp.linalg.eig(
        K_approx, left=True, right=True, check_finite=False
    )

    scal = np.diag(lvecs.conj().T @ rvecs).conj()
    efuncs = ((tempvecs * np.sqrt(tempvals)) @ rvecs) / scal
    lvecs /= scal
    modes = (lvecs.conj().T @ scal_tempvecs.T) * scal[:, np.newaxis]

    res = np.linalg.norm(
        A.conj().T @ modes.conj().T - G @ modes.conj().T * evals.conj(), axis=0
    )
    perm = res.argsort()
    evals, res, efuncs, modes = (
        evals[perm],
        res[perm],
        efuncs[:, perm],
        modes[perm],
    )

    if return_modes:
        modes = modes @ data[:-1]

    if rank is not None:
        _rank = min(cutoff, rank)
        return evals[:_rank], efuncs[:_rank], modes[:_rank], res[:_rank]
    return evals, efuncs, modes, res


def pivchol_kdmd(
    data: np.ndarray,
    kernel: KernelWrapper,
    samples: int,
    rank: int | None = None,
    return_modes: bool = False,
    reg: float = 1e-16,
    reorth: int = 2,
    piv_kind: str = "rp",
):
    if rank:
        rank = int(rank)
    samples, reorth, return_modes = int(samples), int(reorth), bool(return_modes)
    piv_kind, reg = str(piv_kind), float(reg)

    _factor, pivots, _ = piv_cholesky(
        data, kernel, samples=samples, tol=1e-16, kind=piv_kind
    )
    factor = _factor[:, :-1]
    Kmat = _factor.conj().T @ _factor
    G, A = Kmat[:-1, :-1], Kmat[1:, :-1]
    pivots = np.array(pivots, dtype=int)
    mod_pivs = pivots[pivots < data.shape[0] - 1]
    tempvals, tempvecs = sp.linalg.eigh(factor @ factor.T, check_finite=False)
    tempvecs = factor.T @ tempvecs
    tempvecs /= np.linalg.norm(tempvecs, axis=0)

    # orthonormalise tempvecs
    for _ in range(reorth):
        for index in range(1, tempvecs.shape[1]):
            vec = tempvecs[:, index]
            tempvecs[:, index] -= tempvecs[:, :index] @ (tempvecs[:, :index].T @ vec)
            tempvecs[:, index] /= np.linalg.norm(vec)

    perm = tempvals.argsort()[::-1]
    tempvals, tempvecs = tempvals[perm], tempvecs[:, perm]
    cutoff = np.argwhere(tempvals >= tempvals[0] * reg)[-1, 0]
    tempvals, tempvecs = tempvals[:cutoff], tempvecs[:, :cutoff]
    scal_tempvecs = tempvecs / np.sqrt(tempvals)

    K_approx = scal_tempvecs.conj().T @ A @ scal_tempvecs
    evals, lvecs, rvecs = sp.linalg.eig(
        K_approx, left=True, right=True, check_finite=False
    )

    scal = np.diag(lvecs.conj().T @ rvecs).conj()
    efuncs = ((tempvecs * np.sqrt(tempvals)) @ rvecs) / scal
    lvecs /= scal
    modes = (lvecs.conj().T @ scal_tempvecs.T) * scal[:, np.newaxis]

    res = np.linalg.norm(
        A[:, mod_pivs].conj().T @ modes.conj().T
        - G[:, mod_pivs].conj().T @ modes.conj().T * evals.conj(),
        axis=0,
    )
    perm = res.argsort()
    evals, res, efuncs, modes = (
        evals[perm],
        res[perm],
        efuncs[:, perm],
        modes[perm],
    )

    if return_modes:
        modes = modes @ data[:-1]

    if rank is not None:
        _rank = min(cutoff, rank)
        return evals[:_rank], efuncs[:_rank], modes[:_rank], res[:_rank]
    return evals, efuncs, modes, res


def oasis_kdmd(
    data: np.ndarray,
    kernel: KernelWrapper,
    samples: int,
    rank: int | None = None,
    return_modes: bool = False,
    reg: float = 1e-16,
    reorth: int = 2,
):
    if rank:
        rank = int(rank)
    samples, reorth, return_modes = int(samples), int(reorth), bool(return_modes)
    reg = float(reg)

    kernel_mat = kernel(data)
    A = kernel_mat[1:, :-1]

    cols, W, pivots = oasis(
        data,
        kernel,
        samples=samples,
        tol=1e-16,
        initial_samples=1,
    )
    tempvals, tempvecs = sp.linalg.eigh(W, check_finite=False)
    perm = tempvals.argsort()[::-1]
    tempvals, tempvecs = tempvals[perm], tempvecs[:, perm]

    cutoff = np.argwhere(tempvals >= tempvals[0] * reg)[-1, 0]
    tempvals, tempvecs = tempvals[:cutoff], tempvecs[:, :cutoff]
    tempvecs = np.sqrt(cutoff / (data.shape[0] - 1)) * cols.T @ tempvecs / tempvals
    tempvals = ((data.shape[0] - 1) / cutoff) * tempvals
    tempvecs /= np.linalg.norm(tempvecs, axis=0)

    # orthonormalise tempvecs
    for _ in range(reorth):
        for index in range(1, tempvecs.shape[1]):
            vec = tempvecs[:, index]
            tempvecs[:, index] -= tempvecs[:, :index] @ (tempvecs[:, :index].T @ vec)
            tempvecs[:, index] /= np.linalg.norm(vec)

    perm = tempvals.argsort()[::-1]
    tempvals, tempvecs = tempvals[perm], tempvecs[:, perm]

    cutoff = np.argwhere(tempvals >= tempvals[0] * reg)[-1, 0]
    tempvals, tempvecs = tempvals[:cutoff], tempvecs[:, :cutoff]
    scal_tempvecs = tempvecs / np.sqrt(tempvals)
    K_approx = scal_tempvecs.conj().T @ A @ scal_tempvecs
    evals, lvecs, rvecs = sp.linalg.eig(
        K_approx, left=True, right=True, check_finite=False
    )

    scal = np.diag(lvecs.conj().T @ rvecs).conj()
    efuncs = ((tempvecs * np.sqrt(tempvals)) @ rvecs) / scal
    lvecs /= scal
    modes = (lvecs.conj().T @ scal_tempvecs.T) * scal[:, np.newaxis]

    res = np.linalg.norm(
        A[:, pivots].conj().T @ modes.conj().T
        - cols.conj() @ modes.conj().T * evals.conj(),
        axis=0,
    )
    # # flip real and complex part of inner modes
    # _sl = slice(1, -1) if len(modes) % 2 == 0 else slice(1, None)
    # modes[_sl] = -1j * modes[_sl].conj()

    if return_modes:
        modes = modes @ data[:-1]

    if rank is not None:
        _rank = min(cutoff, rank)
        return evals[:_rank], efuncs[:_rank], modes[:_rank], res[:_rank]
    return evals, efuncs, modes, res


def rff_kdmd(
    data: np.ndarray,
    samples: int,
    return_modes: bool = False,
    kernel: tuple[str, tuple, dict] = ("normal", (), {"loc": 0, "scale": 1}),
    reg: float = 1e-16,
    rank: int | None = None,
    mask: None | np.ma.MaskedArray = None,
    stride=1,
    seed: int | None = 2222,
):
    samples, return_modes, reg = int(samples), bool(return_modes), float(reg)
    kernel_func, kargs, kwargs = (
        getattr(np.random, str(kernel[0]).lower()),
        kernel[1],
        kernel[2],
    )
    if rank:
        rank = int(rank)
    assert "size" not in kwargs.keys(), "The keyword 'size' may not be used."
    if seed is not None:
        np.random.seed(int(seed))

    kwargs = kwargs | {
        "size": (
            data.shape[1]
            if mask is None
            else data[0][mask.reshape(*data[0].shape)].shape[0],
            samples,
        )
    }
    zs = kernel_func(*kargs, **kwargs)

    def observable(arr):
        if mask is not None:
            ret = np.zeros((arr.shape[0], samples), dtype=complex)
            _iter = batched(range(arr.shape[0]), n=stride)
            for batch in tqdm(
                _iter,
                desc="Assembling RFF observables",
                total=arr.shape[0] // stride + 1,
            ):
                inds = slice(batch[0], batch[-1] + 1)
                aa = arr[inds]
                ret[inds] = np.exp(1j * np.dot(aa[:, mask.reshape(*aa.shape[1:])], zs))
            return ret
        return np.exp(1j * arr @ zs)

    obs = observable(data)
    oX, oY = obs[:-1], obs[1:]
    U, S, V = sp.linalg.svd(oX, full_matrices=False, check_finite=False)
    cutoff = np.argwhere(S >= S[0] * reg)[-1, 0]
    U, S, V = U[:, :cutoff], S[:cutoff], V[:cutoff]
    K_approx = ((U / S).conj().T @ oY) @ V.conj().T
    evals, lvecs, rvecs = sp.linalg.eig(
        K_approx, left=True, right=True, check_finite=False
    )

    # # equivalent to U * S @ rvecs
    # efuncs = oX @ V.conj().T @ rvecs
    # # scale left eigvecs such that v_l[:, i].conj().T @ v_r[:, j] = delta_ij
    # # lvecs /= np.diag(lvecs.conj().T @ rvecs).conj()
    # modes = lvecs.conj().T @ (U / S).conj().T

    scal = np.diag(lvecs.conj().T @ rvecs).conj()
    efuncs = (oX @ V.conj().T @ rvecs) / scal
    lvecs /= scal
    modes = (lvecs.conj().T @ (U / S).conj().T) * scal[:, np.newaxis]

    res = np.linalg.norm(
        lvecs.conj().T @ (U / S).conj().T @ oY
        - lvecs.conj().T @ V * evals[:, np.newaxis],
        axis=1,
    )
    perm = res.argsort()
    evals, res, efuncs, modes = (
        evals[perm],
        res[perm],
        efuncs[:, perm],
        modes[perm],
    )

    if return_modes:
        modes = modes @ data[:-1]

    if rank is not None:
        _rank = min(cutoff, rank)
        return evals[:_rank], efuncs[:_rank], modes[:_rank], res[:_rank]
    return evals, efuncs, modes, res
