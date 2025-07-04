import os
from itertools import batched

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as _kernel
from tqdm import tqdm

from gndmd import pivchol_kdmd, rff_kdmd
from gndmd.utils import KernelWrapper


def main():
    stride = 500
    data = netCDF4.Dataset("DATA/sst.week.mean.nc")
    lat, lon = data["lat"][:], data["lon"][:]
    mode_shape = (len(lat), len(lon))
    data = np.asanyarray(data["sst"])
    # get the mask for the continents
    # the netCDF file provides a valid range of values of [-3, 45]
    mask = data[0] > -3
    np.random.default_rng(1111)
    print(f"Data shape is (NT, STATEDIM): {(data.shape[0], np.prod(data.shape[1:]))}")

    # setup methods to call
    kernel = KernelWrapper(_kernel, mask=mask, stride=stride)
    funcs = (pivchol_kdmd, rff_kdmd)
    func_args = (
        {"kernel": kernel},
        {
            "kernel": ("normal", (), {"scale": 1 / 100}),
            "mask": mask,
            "stride": stride,
        },
    )
    func_names = ("RP_Cholesky", "RFF")

    # run experiments and save bare modes
    samples = [25, 50, 75, 100, 125, 150, 175, 200]
    for func, fargs, fname in zip(funcs, func_args, func_names):
        _modes = {}
        for _samples in samples:
            _, _, modes, _ = func(data, samples=_samples, **fargs)  # pyright: ignore
            _modes[str(_samples)] = modes
        np.savez_compressed(f"sst_modes_{fname}.npz", **_modes)

    # read bare mode files, assemble modes, and plot
    n_modes = 6
    for fname in [f for f in os.listdir(".") if os.path.isfile(f)]:
        if fname.startswith("sst_modes"):
            method = fname.split(".")[0].split("_")[-1]
            print(method)
            _modes = np.load(fname, allow_pickle=True)
            for _samples, _mode in _modes.items():
                temp_mode = np.zeros(
                    (_mode.shape[0], len(data[0, mask])), dtype=complex
                )
                # assemble all modes in chunks
                for rnge in tqdm(
                    batched(range(data.shape[0] - 1), n=stride),
                    desc="Assembling modes",
                    total=(data.shape[0] - 1) // stride + 1,
                ):
                    inds = slice(rnge[0], rnge[-1] + 1)
                    dd = data[inds]
                    temp_mode += np.dot(_mode[:, inds], dd[:, mask])

                # create plots
                _, axs = plt.subplots(
                    ncols=2, nrows=3, num=f"Modes for {method} {_samples} samples"
                )
                for index in range(n_modes):
                    cind, rind = index % 2, index // 2
                    _ax = axs[rind, cind]
                    plt_modes = np.full((temp_mode.shape[0], *mode_shape), np.nan)
                    plt_modes[:, mask] = temp_mode.real
                    _ax.imshow(np.flipud(plt_modes[2 * index].real), cmap="RdYlBu")
                plt.tight_layout()
                plt.savefig(f"plot_sst_modes_{method}_{_samples}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
