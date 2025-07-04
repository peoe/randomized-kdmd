# MIT License

# Copyright (c) 2022 Ethan Epperly

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from time import time
from warnings import warn

import numpy as np
import scipy as sp

greedy_lapack = sp.linalg.get_lapack_funcs("pstrf", dtype=np.float64)


def cholesky_helper(
    A: np.ndarray, k: int, alg: str = "rp", stoptol: float = 0
) -> tuple[np.ndarray, list, np.ndarray]:
    stoptol, alg = float(stoptol), str(alg)

    n = A.shape[0]
    diags = np.diag(A)
    orig_trace = diags.sum()

    # row ordering, is much faster for large scale problems
    G = np.zeros((k, n))
    rows = np.zeros((k, n))
    rng = np.random.default_rng()

    arr_idx = []
    for i in range(k):
        if alg == "rp":
            idx = rng.choice(range(n), p=diags / sum(diags))
        elif alg == "rgreedy":
            idx = rng.choice(np.where(diags == np.max(diags))[0])
        elif alg == "greedy":
            idx = np.argmax(diags)
        else:
            raise RuntimeError(f"Algorithm {alg} not recognized")

        arr_idx.append(idx)
        rows[i, :] = A[idx, :]
        G[i, :] = (rows[i, :] - G[:i, idx].T @ G[:i, :]) / np.sqrt(diags[idx])
        diags -= G[i, :] ** 2
        diags = diags.clip(min=0)

        if stoptol > 0 and diags.sum() <= stoptol * orig_trace:
            G = G[:i, :]
            rows = rows[:i, :]
            break

    return G, arr_idx, rows


def _greedy_cholesky(
    A: np.ndarray, atol: float = None, rtol: float = None
) -> tuple[np.ndarray, list, int]:
    atol, rtol = float(atol), float(rtol)

    trace = np.einsum("ii->", A)
    L, piv, rank, info = greedy_lapack(A, lower=True)
    L = np.tril(L)
    if rtol is not None and atol is not None:
        atol = 0 if atol is None else atol
        rtol = 0 if rtol is None else rtol
        trailing_sums = trace - np.cumsum(np.linalg.norm(L, axis=0) ** 2)
        rank = np.argmax(trailing_sums < (atol + rtol * trace)) + 1
    return L, piv, rank


def block_cholesky_helper(
    A: np.ndarray,
    k: int,
    b: int,
    alg: str,
    stoptol: float = 1e-14,
    strategy: str = "regularize",
    rbrp_atol: float = 0.0,
    rbrp_rtol: str = "1/b",
) -> tuple[np.ndarray, list, np.ndarray]:
    b, stoptol, rbrp_atol = int(b), float(stoptol), float(rbrp_atol)
    alg, rbrp_rtol = str(alg), str(rbrp_rtol)

    diags = np.diag(A)
    n = A.shape[0]
    orig_trace = diags.sum()
    scale = 2 * max(diags)
    if "1/b" == rbrp_rtol:
        rbrp_rtol = 1.0 / b

    # row ordering
    G = np.zeros((k, n))
    rows = np.zeros((k, n))

    rng = np.random.default_rng()

    arr_idx = []
    counter = 0
    while counter < k:
        block_size = min(k - counter, b)

        if alg == "rp":
            idx = rng.choice(
                range(n), size=2 * block_size, p=diags / diags.sum(), replace=True
            )
            idx = np.unique(idx)[:block_size]
            block_size = len(idx)
        elif alg == "greedy":
            idx = np.argpartition(diags, -block_size)[-block_size:]
        else:
            raise RuntimeError(f"Algorithm {alg} not recognized")

        if strategy in ("regularize", "regularized"):
            arr_idx.extend(idx)
            rows[counter : counter + block_size, :] = A[idx, :]
            G[counter : counter + block_size, :] = (
                rows[counter : counter + block_size, :]
                - G[0:counter, idx].T @ G[0:counter, :]
            )
            C = G[counter : counter + block_size, idx]

            try:
                L = np.linalg.cholesky(
                    C + np.finfo(float).eps * b * scale * np.identity(block_size)
                )
                G[counter : counter + block_size, :] = np.linalg.solve(
                    L, G[counter : counter + block_size, :]
                )
            except np.linalg.LinAlgError:
                warn(
                    "Cholesky failed in block partial Cholesky. Falling back to eigendecomposition"
                )
                evals, evecs = np.linalg.eigh(C)
                evals[evals > 0] = evals[evals > 0] ** (-0.5)
                evals[evals < 0] = 0
                G[counter : counter + block_size, :] = evals[:, np.newaxis] * (
                    evecs.T @ G[counter : counter + block_size, :]
                )
        elif strategy in ("rbrp", "pivoting", "pivoted"):
            H = A[idx, idx] - G[0:counter, idx].T @ G[0:counter, idx]
            L, piv, rank = _greedy_cholesky(H, atol=rbrp_atol, rtol=rbrp_rtol)
            idx = idx[piv[0:rank] - 1]
            arr_idx.extend(idx)
            rows[counter : counter + len(idx), :] = A[idx, :]
            G[counter : counter + len(idx), :] = (
                rows[counter : counter + len(idx), :]
                - G[0:counter, idx].T @ G[0:counter, :]
            )
            G[counter : counter + len(idx), :] = np.linalg.solve(
                L[0:rank, 0:rank], G[counter : counter + len(idx), :]
            )
        else:
            raise ValueError(f"{strategy} is not a valid strategy for block RPCholesky")

        diags -= np.sum(G[counter : counter + len(idx), :] ** 2, axis=0)
        diags = diags.clip(min=0)

        counter += len(idx)

        if stoptol > 0 and diags.sum() <= stoptol * orig_trace:
            G = G[:counter, :]
            rows = rows[:counter, :]
            break

    return G, arr_idx, rows


def rejection_cholesky(H: np.ndarray) -> tuple[np.ndarray, list]:
    b = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise RuntimeError("rejection_cholesky requires a square matrix")
    if np.trace(H) <= 0:
        raise RuntimeError("rejection_cholesky requires a strictly positive trace")
    u = np.array([H[j, j] for j in range(b)])

    idx = []
    L = np.zeros((b, b))
    for j in range(b):
        if np.random.rand() * u[j] < H[j, j]:
            idx.append(j)
            L[j:, j] = H[j:, j] / np.sqrt(H[j, j])
            H[j + 1 :, j + 1 :] -= np.outer(L[j + 1 :, j], L[j + 1 :, j])
    idx = np.array(idx)
    L = L[np.ix_(idx, idx)]
    return L, idx


def accelerated_rpcholesky(
    A: np.ndarray,
    k: float,
    b: str | int = "auto",
    stoptol: float = 1e-13,
    verbose: bool = False,
) -> tuple[np.ndarray, list, np.ndarray]:
    stoptol = float(stoptol)
    b, verbose = b if isinstance(b, str) else int(b), bool(verbose)
    auto_b = False
    if isinstance(b, str) and "auto" == b:
        b = int(np.ceil(k / 10))
        auto_b = True

    diags = A.diag()
    n = A.shape[0]
    orig_trace = diags.sum()

    # row ordering
    G = np.zeros((k, n))
    rows = np.zeros((k, n))
    rng = np.random.default_rng()
    arr_idx = np.zeros(k, dtype=int)
    counter = 0
    while counter < k:
        idx = rng.choice(range(n), size=b, p=diags / diags.sum(), replace=True)

        if auto_b:
            start = time()

        H = A[idx, idx] - G[0:counter, idx].T @ G[0:counter, idx]
        L, accepted = rejection_cholesky(H)
        num_sel = len(accepted)

        if num_sel > k - counter:
            num_sel = k - counter
            accepted = accepted[:num_sel]
            L = L[:num_sel, :num_sel]

        idx = idx[accepted]

        if auto_b:
            rejection_time = time() - start
            start = time()

        arr_idx[counter : counter + num_sel] = idx
        rows[counter : counter + num_sel, :] = A[idx, :]
        G[counter : counter + num_sel, :] = (
            rows[counter : counter + num_sel, :] - G[0:counter, idx].T @ G[0:counter, :]
        )
        G[counter : counter + num_sel, :] = np.linalg.solve(
            L, G[counter : counter + num_sel, :]
        )
        diags -= np.sum(G[counter : counter + num_sel, :] ** 2, axis=0)
        diags = diags.clip(min=0)

        if auto_b:
            process_time = time() - start

            # Assuming rejection_time ~ A b^2 and process_time ~ C b
            # then obtaining rejection_time = process_time / 4 entails
            # b = C / 4A = (process_time / b) / 4 (rejection_time / b^2)
            #   = b * process_time / (4 * rejection_time)
            target = int(np.ceil(b * process_time / (4 * rejection_time)))
            b = max(
                [
                    min([target, int(np.ceil(1.5 * b)), int(np.ceil(k / 3))]),
                    int(np.ceil(b / 3)),
                    10,
                ]
            )

        counter += num_sel

        if stoptol > 0 and diags.sum() <= stoptol * orig_trace:
            G = G[:counter, :]
            rows = rows[:counter, :]
            break

        if verbose:
            print(f"Accepted {num_sel} / {b}")

    return G, arr_idx, rows


def rpcholesky(
    A: np.ndarray, k: int, b: int | None = None, accelerated: bool = True, **kwargs
) -> tuple[np.ndarray, list, np.ndarray]:
    k = int(k)
    A = np.asanyarray(A)

    if b is None:
        if accelerated:
            return accelerated_rpcholesky(A=A, k=k, **kwargs)
        return cholesky_helper(A=A, k=k, alg="rp", **kwargs)
    if accelerated:
        return accelerated_rpcholesky(A=A, k=k, b=b, **kwargs)
    return block_cholesky_helper(A=A, k=k, b=b, alg="rp", **kwargs)


def greedy(
    A: np.ndarray, k: int, randomized_tiebreaking: bool = False, b: int = 1, **kwargs
) -> tuple[np.ndarray, list, np.ndarray]:
    k, b = int(k), int(b)
    randomized_tiebreaking = bool(randomized_tiebreaking)
    A = np.asanyarray(A)

    if b == 1:
        return cholesky_helper(
            A=A, k=k, alg="rgreedy" if randomized_tiebreaking else "greedy", **kwargs
        )
    if randomized_tiebreaking:
        warn("Randomized tiebreaking not implemented for block greedy method")
    return block_cholesky_helper(A=A, k=k, b=b, alg="greedy", **kwargs)
