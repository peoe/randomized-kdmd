from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as _kernel
from tqdm import tqdm

from gndmd import kdmd, oasis_kdmd, pivchol_kdmd, rff_kdmd
from gndmd.utils import KernelWrapper


def main():
    data = np.load("DATA/duffing.npy")
    data = data.T
    data = np.repeat(data, 5000, axis=1)
    np.random.default_rng(1111)
    print(f"Data shape is (NT, STATEDIM): {data.shape}")

    # setup methods to call
    kernel = KernelWrapper(_kernel)
    oasis_kernel = KernelWrapper(_kernel, offset=1)
    funcs = (kdmd, rff_kdmd, oasis_kdmd, pivchol_kdmd)
    func_args = (
        {"kernel": kernel, "return_modes": True},
        {"kernel": ("normal", (), {"scale": 1 / 20}), "return_modes": True},
        {"kernel": oasis_kernel, "return_modes": True},
        {"kernel": kernel, "return_modes": True},
    )
    func_names = ("KDMD", "RFF", "oASIS", "Algorithm 2")

    # run experiments
    samples = np.linspace(10, 150, num=15, dtype=int)
    ref_time, ref_err = 0, 0
    errs, times = {}, {}
    for func, fargs, fname in tqdm(zip(funcs, func_args, func_names)):
        if fname == "KDMD":
            tic = time()
            _, efuncs, modes, _ = func(data, **fargs)  # pyright: ignore
            toc = time()
            recon = (efuncs @ modes).real
            ref_time = toc - tic
            ref_err = np.linalg.norm(data[:-1] - recon, axis=1) / np.linalg.norm(
                data[:-1], axis=1
            )
            ref_err = ((ref_err * np.ones(ref_err.shape)) ** 2).sum()
            continue
        _errs, _times = [], []
        for _samples in samples:
            tic = time()
            _, efuncs, modes, _ = func(data, samples=_samples, **fargs)  # pyright: ignore
            toc = time()
            recon = (efuncs @ modes).real
            err = np.linalg.norm(data[:-1] - recon, axis=1)
            err /= np.linalg.norm(data[:-1], axis=1)
            _errs.append(((err * np.ones(err.shape)) ** 2).sum())
            _times.append(toc - tic)
        errs[fname] = _errs
        if fname != "RFF":
            times[fname] = _times

    # plot errors
    plt.figure(num="Errors")
    plt.semilogy(samples, ref_err * np.ones(len(samples)), label="KDMD", marker="x")
    for fname, err in errs.items():
        plt.semilogy(samples, err, label=fname, marker="x")
    plt.xlabel(r"Number of samples $S$")
    plt.ylabel("Relative error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("duffing_errors.pdf")

    # plot times
    width = 0.15
    samples, inds = [50, 100, 150], [4, 9, 14]
    pos = np.arange(len(samples))  # pyright: ignore
    plt.figure(num="Computational times")
    max_t = ref_time
    plt.bar(pos, ref_time * np.ones(len(samples)), width, label="KDMD")
    for index, (method, ts) in enumerate(times.items(), start=1):
        ts = np.array(ts)
        offset = width * index
        plt.bar(pos + offset, ts[inds], width, label=method)
        max_t = max(max_t, np.max(ts[inds]))
    plt.xticks(pos + width, samples)  # pyright: ignore
    plt.xlabel(r"Number of samples $S$")
    plt.ylabel("Elapsed computational time (s)")
    plt.ylim(0, max_t * 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("duffing_times.pdf")
    plt.show()


if __name__ == "__main__":
    main()
