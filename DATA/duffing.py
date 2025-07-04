import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm


def main():
    num_trajs, traj_length = 700, 11
    delta = 0.5
    t_span, plot_t_span = (0, 25), (0, 25)
    ts, plot_ts = np.linspace(*t_span, traj_length), np.linspace(*plot_t_span, 1000)
    inits = np.random.uniform(-2, 2, size=(num_trajs, 2))
    a1, a2 = np.array([[-1, 0], [1, 0]])
    tol = 1e-2

    def duffing(_, state):
        x, xdot = state
        return np.stack([xdot, -delta * xdot - x * (x**2 - 1)])

    data = None
    plt.figure(figsize=(10, 10))
    for init in tqdm(inits):
        eval = solve_ivp(
            duffing, plot_t_span, init, dense_output=True, vectorized=True
        )["sol"]
        data = np.append(data, eval(ts), axis=1) if data is not None else eval(ts)
        traj = eval(plot_ts)
        color = (
            "b"
            if np.linalg.norm(traj[:, -1] - a1) < tol
            else ("r" if np.linalg.norm(traj[:, -1] - a2) < tol else "g")
        )
        plt.scatter(traj[0], traj[1], s=3, c=color, alpha=0.3)
    print(data.shape)
    plt.gca().set_xlim([-2, 2])
    plt.gca().set_ylim([-2, 2])
    np.save("DATA/duffing.npy", data)
    plt.show()


if __name__ == "__main__":
    main()
