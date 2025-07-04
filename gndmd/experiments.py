from collections.abc import Callable
from itertools import batched, product
from time import perf_counter as time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .utils import reconstruction_coeffs


class Experiment:
    def __init__(
        self,
        ranks: list[int],
        data: np.ndarray,
        funcs: tuple[Callable],
        func_args: tuple[dict],
        func_names: tuple[str],
        normalize: bool = True,
        error_metrics: dict = {
            "max": None,
            "uniform": lambda z: np.ones(len(z)),
        },
    ):
        self._data = np.asanyarray(data)
        normalize = bool(normalize)

        self.ncdf = False
        self.data = self._data
        if isinstance(self.data, np.ma.MaskedArray):
            # for netCDF files we have shape[0] == num snapshots
            M = self.data.shape[0]
            self.data_scale = np.ones(M)
            self.ncdf = True
            print(f"Data shape is (NT, STATEDIM): {(M, np.prod(self.data.shape[1:]))}")
            self.X, self.Y = None, None
        else:
            self.data_scale = np.ones(self._data.shape[1])
            if normalize:
                self.data_scale = np.linalg.norm(self._data, axis=0)
                self.data /= self.data_scale
            self.data = self.data.T
            print(f"Data shape is (NT, STATEDIM): {self.data.shape}")
            self.X, self.Y = self.data[:-1], self.data[1:]

        assert len(funcs) == len(func_args) == len(func_names)
        descs = tuple(map(lambda s: str(s).lower(), func_names))
        for func_arg in func_args:
            # banned func_args
            # we want to handle these on our own
            assert "rank" not in func_arg
            assert "return_modes" not in func_arg
        self.func_ids = {
            index: (f, fargs, desc)
            for index, (f, fargs, desc) in enumerate(zip(funcs, func_args, descs))
        }
        self.ranks = list(map(int, ranks))

        for metric in error_metrics.values():
            assert metric is None or callable(metric)
        self.error_metrics = error_metrics

        self._run = False
        self._cache = {
            rank: {func_id: None for func_id in self.func_ids.keys()}
            for rank in self.ranks
        }

    def run(self, rank: None | int = None):
        if rank and rank not in self._cache:
            pass
        elif not self._run:  # only run if no previous data exists
            for rank, id in tqdm(
                product(self.ranks, self.func_ids.keys()),
                total=len(self.ranks) * len(self.func_ids),
                desc="Running experiments",
            ):
                self._cache[rank][id] = self._call(rank, id)  # pyright: ignore

            self._run = True

    def _call(self, rank: int, id: int):
        func, args, _ = self.func_ids[id]
        args = args | {"rank": rank}

        tic = time()
        evals, efuncs, modes, res = func(self.data, **args)
        toc = time()

        coeffs, metrics = None, None
        if len(self.error_metrics) > 0:
            recon, coeffs = self.reconstruct(id, evals, efuncs, modes)
            metrics = self._metrics(id, recon)

        return {
            "desc": self.func_ids[id][2],
            "evals": evals,
            "efuncs": efuncs,
            "modes": modes,
            "res": res,
            "coeffs": coeffs,
            "metrics": metrics,
            "time": toc - tic,
        }

    def reconstruct(
        self,
        id: int,
        evals: None | np.ndarray = None,
        efuncs: None | np.ndarray = None,
        modes: None | np.ndarray = None,
        rank: None | int = None,
    ) -> tuple[dict, np.ndarray | None]:
        id = int(id)
        assert id in self.func_ids.keys()

        if rank:
            rank = int(rank)
            assert rank in self.ranks

            evals = self.cache[rank][id]["evals"]
            efuncs = self.cache[rank][id]["efuncs"]
            modes = self.cache[rank][id]["modes"]

        assert evals is not None and modes is not None

        vander = np.vander(evals, N=len(self.data) - 1, increasing=True)  # pyright: ignore

        if self.func_ids[id][2] in ("dmd"):
            # reconstruction using normal equations
            coeffs = reconstruction_coeffs(evals, modes, data=self.X)
            recon = (modes[: len(coeffs)].T * coeffs @ vander).real.T  # pyright: ignore
            return recon, coeffs

        modes = modes @ self.X
        coeffs = None
        # reconstruction from efunc times mode product
        recon = (efuncs @ modes).real  # pyright: ignore

        return recon, coeffs

    def _metrics(self, id: int, recon: dict):
        id = int(id)
        assert id in self.func_ids.keys()

        def compute_errs(recon):
            err = np.linalg.norm(((self.X - recon).T * self.data_scale[:-1]).T, axis=1)
            err /= np.linalg.norm(self.X, axis=1)  # pyright: ignore
            errs = {}
            for name, metric in self.error_metrics.items():
                if name == "max":
                    errs[name] = err.max()
                elif name == "min":
                    errs[name] = err.min()
                elif name == "mean":
                    errs[name] = err.mean()
                elif name == "median":
                    errs[name] = np.median(err)
                else:
                    errs[name] = ((err * metric(self.X)) ** 2).sum()
            return errs

        return compute_errs(recon)

    @property
    def times(self):
        assert self._run
        _times = {
            funcname: [self._cache[rank][id]["time"] for rank in self.ranks]  # pyright: ignore
            for id, (_, _, funcname) in self.func_ids.items()
        }
        return _times

    @property
    def errors(self):
        assert self._run
        _errors = {
            funcname: {
                metric: [
                    self._cache[rank][id]["metrics"][metric]  # pyright: ignore
                    for rank in self.ranks
                ]
                for metric in self.error_metrics.keys()
            }  # pyright: ignore
            for id, (_, _, funcname) in self.func_ids.items()
        }
        return _errors

    def plot_times(self):
        assert self._run
        plt.figure()
        plt.title("Computational times in seconds")
        for name, times in self.times.items():
            plt.semilogy(self.ranks, times, label=name)
        plt.xlabel(r"Rank $r$")
        plt.ylabel("Elapsed time")
        plt.legend()

    def bar_chart_times(
        self,
        width: float = 0.1,
        file_name: None | str = None,
    ):
        assert self._run
        width = float(width)
        if file_name is not None:
            file_name = str(file_name)
        fig, ax = plt.subplots(layout="tight")
        if file_name is None:
            fig.suptitle("Elapsed computational times")
        max_t = -1
        pos = np.arange(len(self.times[list(self.times.keys())[0]]))  # pyright: ignore
        for index, (method, ts) in enumerate(self.times.items()):
            offset = width * index
            ax.bar(pos + offset, ts, width, label=method)
            max_t = max(max_t, np.max(ts))
        ax.set_xticks(pos + width, self.ranks)  # pyright: ignore
        ax.set_ylabel("Elapsed computational time (s)")
        ax.set_ylim(0, max_t * 1.05)
        ax.legend()
        if file_name is not None:
            plt.savefig(f"{file_name}.eps", dpi=150)

    def plot_errors(self, file_name: None | str = None):
        assert self._run
        if file_name is not None:
            file_name = str(file_name)
        errors = self.errors
        metrics = self.error_metrics.keys()
        plt.figure()
        if file_name is None:
            plt.title("Error metrics")
        for fname in errors.keys():
            for metric in metrics:
                plt.semilogy(
                    self.ranks, errors[fname][metric], label=f"{fname} {metric}"
                )
        plt.xlabel(r"Rank $r$")
        plt.ylabel(r"Error")
        plt.legend()
        if file_name is not None:
            plt.savefig(f"{file_name}.eps", dpi=150)

    def plot_residuals(self, file_name: None | str = None):
        assert self._run
        if file_name is not None:
            file_name = str(file_name)
        for rank in self.ranks:
            plt.figure()
            if file_name is None:
                plt.title(f"Residuals for rank r = {rank}")
            for fid in self.func_ids.keys():
                plt.semilogy(self._cache[rank][fid]["res"], label=self.func_ids[fid][2])  # pyright: ignore
            plt.xlabel("Index")
            plt.ylabel("Residual")
            plt.legend()
            if file_name is not None:
                plt.savefig(f"{file_name}_{rank}.eps", dpi=150)

    def scatter_evals(self):
        assert self._run
        for rank in self.ranks:
            plt.figure()
            plt.title(f"Eigenvalues for rank r = {rank}")
            for fid in self.func_ids.keys():
                evals = self._cache[rank][fid]["evals"]  # pyright: ignore
                plt.scatter(
                    evals.real,
                    evals.imag,
                    s=25,
                    marker="x",
                    alpha=0.4,
                    label=self.func_ids[fid][2],
                )
            plt.xlabel(r"Re$(\lambda_i)$")
            plt.ylabel(r"Im$(\lambda_i)$")
            plt.legend()

    def plot_modes(
        self,
        mode_shape: tuple[int, int],
        num_modes: int = 5,
        file_name: None | str = None,
    ):
        assert self._run
        assert len(mode_shape) == 2
        mode_shape = tuple(map(int, mode_shape))  # pyright: ignore
        if file_name is not None:
            file_name = str(file_name)
        num_modes = int(num_modes)
        for rank in self.ranks:
            for fid in self.func_ids.keys():
                fig, axs = plt.subplots(ncols=num_modes, figsize=(5 * num_modes, 3))  # pyright: ignore
                method = self.func_ids[fid][2]
                if file_name is None:
                    fig.suptitle(f"{method} modes")
                for index in range(num_modes):
                    _modes = self._cache[rank][fid]["modes"]  # pyright: ignore
                    if method != "dmd":
                        _modes = _modes @ self.X
                    _p = _modes[2 * index].real.reshape(mode_shape)
                    axs[index].imshow(_p.T)
                    axs[index].set_xlabel("x")
                    axs[index].set_ylabel("y")
                plt.tight_layout()
                if file_name is not None:
                    plt.savefig(f"{file_name}_{method}_{rank}.eps", dpi=150)

    def store_modes(
        self,
        file_name: str,
        mask: None | np.ma.MaskedArray = None,
        stride: int = 1,
    ):
        assert self._run
        file_name = str(file_name)
        stride = int(stride)
        for rank in self.ranks:
            for fid in self.func_ids.keys():
                method = self.func_ids[fid][2]
                _modes = self._cache[rank][fid]["modes"]  # pyright: ignore
                if method != "dmd":
                    if mask is not None:
                        temp_modes = np.zeros(
                            (_modes.shape[0], len(self.data[0, mask])), dtype=complex
                        )
                        # assemble all modes in chunks
                        for rnge in tqdm(
                            batched(range(self.data.shape[0] - 1), n=stride),
                            desc="Assembling modes",
                            total=(self.data.shape[0] - 1) // stride + 1,
                        ):
                            inds = slice(rnge[0], rnge[-1] + 1)
                            dd = self.data[inds]
                            temp_modes += np.dot(_modes[:, inds], dd[:, mask])
                        _modes = temp_modes
                    else:
                        _modes = _modes @ self.X
                np.savez_compressed(f"{file_name}_{method}_{rank}.npz", modes=_modes)

    def plot_modes_from_file(
        self,
        file_name: str,
        mode_shape: tuple,
        transpose: bool = True,
        flipud: bool = False,
        cmap: str = "RdYlBu",
        mask: None | np.ma.MaskedArray = None,
    ):
        file_name, cmap = str(file_name), str(cmap)
        mode_shape = tuple(map(int, mode_shape))
        transpose, flipud = bool(transpose), bool(flipud)
        try:
            file_content = np.load(file_name)
        except OSError:
            print(f"File not found: {file_name}")
            return

        method, rank = file_name.split(".")[0].split("_")[-2:]

        _modes = file_content["modes"]
        n_modes = len(_modes)
        if mask is not None:
            modes = np.full((n_modes, *mode_shape), np.nan)
            modes[:, mask] = _modes.real
        else:
            modes = _modes.reshape((n_modes, *mode_shape)).real

        t1 = lambda arr: arr.T if transpose else arr  # noqa
        t2 = lambda arr: np.flipud(arr) if flipud else arr  # noqa
        transform = lambda arr: t1(t2(arr))  # noqa

        n_cols = int(np.ceil(np.sqrt(n_modes)))
        n_rows = int(np.ceil(n_modes / n_cols))
        fig_shape = (n_rows, n_cols)

        fig, axs = plt.subplots(*fig_shape, sharex=True, sharey=True)
        fig.suptitle(f"Modes for {method} rank {rank}")
        for ind, mode in enumerate(modes):
            rind, cind = ind % n_cols, ind // n_rows
            ax = axs[rind, cind]
            ax.imshow(transform(mode.real), cmap=cmap)

    def __repr__(self):
        return f"Experiment for ranks {self.ranks} and functions {[fname for _, _, fname in self.func_ids.values()]}"


class DataWrapper:
    def __init__(self, data, name: str):
        self._data = data
        self._name = str(name)

    @property
    def shape(self):
        return self._data.shape

    def __getitem__(self, index: tuple | int | slice | np.ndarray):
        if self._name == "Y":
            pass
        ret = self._data[index]
        _s = ret.shape[0]
        return ret.reshape(_s, -1)

    def __len__(self):
        return self._data.shape[0] - 1
