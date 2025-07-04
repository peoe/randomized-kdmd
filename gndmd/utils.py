from collections.abc import Callable
from itertools import batched, combinations_with_replacement, product

import numpy as np
import scipy as sp
from tqdm import tqdm


class KernelWrapper:
    def __init__(
        self,
        kernel: Callable,
        mask: None | np.ma.MaskedArray = None,
        stride: None | int = None,
        offset: int = 0,
    ):
        self._kernel = kernel
        self.mask = mask
        self.stride = 1
        self.offset = int(offset)
        if stride is not None:
            self.stride = int(stride)

    def __call__(self, X, Y=None, trunc_X: bool = False):
        # trunc_X controls if we wish to remove the last index of the 0th axis from the X array
        trunc_X = bool(trunc_X)
        if isinstance(X, np.ma.MaskedArray):
            sym = False
            if Y is None:
                Y = X
                sym = True

            _x, _y = 1, 1
            if len(X.shape) > 2:
                _x = X.shape[0] if not trunc_X else X.shape[0] - self.offset
            else:
                X = X.reshape(1, -1)
            if len(Y.shape) > 2:
                _y = Y.shape[0] if not trunc_X else Y.shape[0] - self.offset
            else:
                Y = Y.reshape(1, -1)

            ret = np.zeros((_x, _y))
            x_iter = batched(range(_x), n=self.stride if self.stride is not None else 1)
            if sym:
                _iter = combinations_with_replacement(x_iter, 2)  # pyright: ignore
                _total = ((_x // self.stride + 1) ** 2 + (_x // self.stride + 1)) / 2
            else:
                y_iter = batched(
                    range(_y), n=self.stride if self.stride is not None else 1
                )
                _iter = product(x_iter, y_iter)  # pyright: ignore
                _total = (_x // self.stride + 1) ** 2
            if _total > 5:
                _iter = tqdm(_iter, desc="Computing the kernel", total=int(_total))
            for xinds, yinds in _iter:
                x_slice, y_slice = (
                    slice(xinds[0], xinds[-1] + 1),
                    slice(yinds[0], yinds[-1] + 1),
                )
                xx, yy = X[x_slice], Y[y_slice]
                if self.mask is not None:
                    xx, yy = (
                        xx[:, self.mask.reshape(*xx.shape[1:])],
                        yy[:, self.mask.reshape(*yy.shape[1:])],
                    )
                xx = xx.reshape(len(xinds), -1)
                yy = yy.reshape(len(yinds), -1)
                ret[x_slice, y_slice] = self._kernel(xx, yy)
                if sym:
                    ret[y_slice, x_slice] = ret[x_slice, y_slice].T
            return ret

        if Y is None:
            Y = X

        _x, _y = 1, 1
        if len(X.shape) > 1:
            _x = X.shape[0] if not trunc_X else X.shape[0] - self.offset
        else:
            X = X.reshape(1, -1)
        if len(Y.shape) > 1:
            _y = Y.shape[0] if not trunc_X else Y.shape[0] - self.offset
        else:
            Y = Y.reshape(1, -1)

        return self._kernel(X[:_x], Y[:_y])


def print_err_metrics(X: np.ndarray, recon: np.ndarray, desc: str):
    X, recon = np.asanyarray(X), np.asanyarray(recon)
    desc = str(desc)

    err = np.linalg.norm(X - recon, axis=1)
    err /= np.linalg.norm(X, axis=1)
    print(
        f"{desc} max - {err.max():1.2e} mean - {err.mean():1.2e} median - {np.median(err):1.2e} min - {err.min():1.2e}"
    )


def piv_cholesky(
    X: np.ndarray,
    kernel: KernelWrapper,
    samples: int,
    tol: float = 1e-8,
    kind: str = "rp",
    seed: int | None = 2222,
):
    samples, tol = int(samples), float(tol)
    kind = str(kind)
    if seed is not None:
        np.random.seed(int(seed))

    factor = np.empty((samples, X.shape[0]))
    rows = {}
    diag = np.array([kernel(X[ind])[0, 0] for ind in range(len(X))])
    trace = diag.sum()
    pivots = []

    def get_row(index):
        val = rows.get(index, None)
        if val is None:
            val = kernel(X[index], X, trunc_X=True)[0]
            rows[index] = val
        return val

    select = (  # noqa
        lambda d: np.random.choice(len(d), p=d / d.sum())
        if kind == "rp"
        else np.random.choice(np.where(d == np.max(d))[0])
    )

    for ind in range(samples):
        if tol > 0 and diag.sum() <= tol * trace:
            factor = factor[:ind]
            break

        sample = select(diag)
        pivots.append(sample)
        row = get_row(sample)
        factor[ind] = (row - factor[:ind, sample].T @ factor[:ind]) / np.sqrt(
            diag[sample]
        )
        diag -= factor[ind] ** 2
        diag = np.maximum(diag, 0)

    return factor, pivots, np.stack([rows[ind] for ind in pivots])


def oasis(
    X: np.ndarray,
    kernel: KernelWrapper,
    samples: int,
    tol: float = 1e-8,
    initial_samples: int = 1,
    seed: int | None = 2222,
):
    samples, initial_samples, tol = int(samples), int(initial_samples), float(tol)
    if seed is not None:
        np.random.seed(int(seed))

    samples = min(samples, X.shape[0] - 1)

    diag = np.array([kernel(X[ind])[0, 0] for ind in range(len(X) - 1)])
    # initial pivots
    pivots = [
        pivot
        for pivot in np.random.choice(range(samples), initial_samples, replace=False)
    ]
    # w.r.t. to oASIS matlab implementation we select rows not columns
    # thus our-C = matlab-C'
    C = [kernel(X[pivot], X, trunc_X=True)[0] for pivot in pivots]
    _C = np.stack(C)
    W = _C[:, pivots]
    R = sp.linalg.solve(W, _C, check_finite=False)
    while len(pivots) < samples:
        delta = np.abs((_C * R).sum(axis=0) - diag)
        maxind = np.argmax(delta)
        # # comparison to greedy diagonal pivoting
        if delta[maxind] < tol * diag.sum():
            _C = np.stack(C)
            break
        g = kernel(X[maxind], X, trunc_X=True)[0]
        _g = g[:, np.newaxis]

        # Schur update
        b, d = _g[pivots].conj().T, g[maxind]
        ainvb = R[:, maxind][:, np.newaxis]
        schur = 1 / (d - b @ ainvb)
        brep = b @ R
        R = np.vstack(
            [R + ainvb * schur @ (brep - _g.conj().T), schur * (-brep + _g.conj().T)]
        )

        C.append(g)
        pivots.append(maxind)
        _C = np.stack(C)

    W = _C[:, pivots]
    return _C, W, pivots


def qr_khatri_rao(evals, R, m):
    binrep = f"{m:b}"[::-1]  # binary representation
    inds = [i for i, ltr in enumerate(binrep) if ltr == "1"]
    local_R, global_R = R.copy(), None
    j, p = 0, 0
    if binrep[0] == 0:
        global_R = local_R
        j, p = 1, 1

    for k in range(len(binrep)):
        _, local_R = np.linalg.qr(np.vstack([local_R, local_R * evals ** (2**k)]))
        if k == inds[j]:
            if global_R is None:
                global_R = local_R
            else:
                _, global_R = np.linalg.qr(np.vstack([global_R, local_R * evals**p]))
            j += 1
            p += 2**k

    return global_R


def qr_reconstruct(evals, modes, data):
    _modes, _data = modes.T, data.T
    m = _data.shape[1]
    vander = np.vander(evals, N=m, increasing=True)
    Q, R = np.linalg.qr(_modes)
    G = Q.conj().T @ _data
    VRGe = (vander.conj() * (R.conj().T @ G)) @ np.ones(m)
    Rs = qr_khatri_rao(evals, R, m)
    alpha = sp.linalg.solve(
        Rs,
        sp.linalg.solve(
            Rs.conj().T,  # pyright: ignore [reportOptionalMemberAccess]
            VRGe,
            assume_a="upper triangular",
            check_finite=False,
        ),
        assume_a="upper triangular",
        check_finite=False,
    )
    partial_recon = G - R * alpha @ vander
    new_rhs = (vander.conj() * (R.conj().T @ partial_recon)) @ np.ones(m)
    correction = sp.linalg.solve(
        Rs,
        sp.linalg.solve(
            Rs.conj().T,  # pyright: ignore [reportOptionalMemberAccess]
            new_rhs,
            assume_a="upper triangular",
            check_finite=False,
        ),
        assume_a="upper triangular",
        check_finite=False,
    )
    coeffs = alpha + correction
    return (_modes[:, : len(coeffs)] * coeffs @ vander[: len(coeffs)]).real.T


def reconstruction_coeffs(evals, modes, data):
    _modes, _data = modes.T, data.T
    m = len(data)
    vander = np.vander(evals, N=m, increasing=True)
    Q, R = np.linalg.qr(_modes)
    G = Q.conj().T @ _data
    coeffs = sp.linalg.solve(
        (R.conj().T @ R) * (vander.conj() @ vander.T),
        (vander.conj() * (R.conj().T @ G)) @ np.ones(m),
        assume_a="pos",
    )

    return coeffs


def reconstruct(evals, modes, data):
    _modes = modes.T
    coeffs = reconstruction_coeffs(evals, modes, data[:-1])
    m = len(data) - 1
    vander = np.vander(evals, N=m, increasing=True)
    return (_modes[:, : len(coeffs)] * coeffs @ vander[: len(coeffs)]).real.T
