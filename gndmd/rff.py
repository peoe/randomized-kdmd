from logging import INFO, basicConfig, getLogger

import numpy as np

FORMAT = "[%(levelname)s] %(name)s: %(message)s"
basicConfig(format=FORMAT)
LOGGER = getLogger("RFF")
LOGGER.setLevel(INFO)


def get_rff(
    n_features: int,
    data: np.ndarray,
    sigma: float = 1.0,
    seed: int | None = None,
    random_mats: tuple[np.ndarray] | None = None,
) -> np.ndarray:
    """
    Compute Random Fourier Features from data.

    Args
    ----
    n_features: the number of features K to use in the approximation
    data:       np.ndarray of shape (dim, num_timesteps)

    Returns
    -------
    rff:        np.ndarray of shape (n_features, num_timesteps)
    """
    n_features = int(n_features)
    data = np.asanyarray(data)
    if seed is not None:
        seed = int(seed)
    dim, nt = data.shape

    if dim <= n_features:
        LOGGER.warning(
            "Data dimension is smaller than requested number of features."
            "Did you pass the data in the format (dim, num_timesteps)?"
        )

    W = np.random.normal(loc=0, scale=1, size=(n_features, dim))
    b = np.random.uniform(0, 2 * np.pi, size=n_features).reshape(-1, 1)

    if random_mats is not None:
        random_mats = tuple(map(np.asanyarray, random_mats))
        assert random_mats[0].shape == (n_features, dim)
        assert random_mats[1].reshape(-1, 1).shape == (n_features, 1)
        W = random_mats[0]
        b = random_mats[1].reshape(-1, 1)

    B = np.repeat(b, nt, axis=1)
    rff = np.sqrt(2) * np.cos(sigma * W @ data + B)
    rff /= np.sqrt(n_features)

    return rff
