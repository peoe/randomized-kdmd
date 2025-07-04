import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics.pairwise import rbf_kernel as _kernel

from gndmd import dmd, kdmd, oasis_kdmd, pivchol_kdmd, rff_kdmd
from gndmd.utils import KernelWrapper


def main():
    mat = loadmat("DATA/CYLINDER_ALL.mat")
    _data = mat["VALL"]
    np.random.default_rng(1111)

    # normalize data
    data_scale = np.ones(_data.shape[1])
    data_scale = np.linalg.norm(_data, axis=0)
    _data /= data_scale
    data = _data.T
    print(f"Data shape is (NT, STATEDIM): {data.shape}")

    # setup methods to call
    kernel = KernelWrapper(_kernel)
    oasis_kernel = KernelWrapper(_kernel, offset=1)
    funcs = (dmd, kdmd, rff_kdmd, oasis_kdmd, pivchol_kdmd)
    func_args = (
        {},
        {"kernel": kernel, "return_modes": True},
        {"kernel": ("normal", (), {"scale": 1 / 10}), "return_modes": True},
        {"kernel": oasis_kernel, "return_modes": True},
        {"kernel": kernel, "return_modes": True},
    )
    func_names = ("DMD", "KDMD", "RFF", "oASIS", "Algorithm 2")

    # run experiments
    n_modes = 3
    samples = 50
    _res = {}
    for func, fargs, fname in zip(funcs, func_args, func_names):
        if fname in ("DMD", "KDMD"):
            _, _, modes, res = func(data, **fargs)  # pyright: ignore
        else:
            _, _, modes, res = func(data, samples=samples, **fargs)  # pyright: ignore
        _res[fname] = res

        # plot modes
        _, axs = plt.subplots(
            ncols=n_modes, figsize=(3 * n_modes, 5), num=f"Modes for {fname}"
        )
        for index in range(n_modes):
            _ax = axs[index]
            _ax.imshow(modes[2 * index].real.reshape(449, 199))
            _ax.set_title(f"Mode {index}\nr({index}) = {res[index]:1.4e}")
            _ax.set_ylabel("x")
            _ax.set_xlabel("y")
        plt.tight_layout()
        plt.savefig(f"fluid_modes_{fname}.pdf")

    # plot residuals
    plt.figure(num="Residuals")
    for fname, res in _res.items():
        plt.semilogy(res, label=fname)
    plt.xlabel(r"Mode index $i$")
    plt.ylabel(r"Residual $r(i)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fluid_residuals.pdf")
    plt.show()


if __name__ == "__main__":
    main()
