# ssh_momentum_space.py

import numpy as np
import matplotlib.pyplot as plt

N = 500       # number of unit cells
v = 1.0       # intracell hopping amplitude
w = 0.8       # intercell hopping amplitude

def ssh_hamiltonian_in_k(N: int, v: float, w: float):
    """
    The momentum-space SSH Hamiltonian with open boundary conditions (C_{N+1}=0).

    Parameters:
    N : Number of unit cells (total sites = 2*N).
    v, w : Hopping amplitudes

    Returns:
    H : The SSH Hamiltonian matrix in real space basis.
    """
    # Brillouin zone in [0, 2π)
    k_vals = np.linspace(-np.pi, np.pi, N)

    # two energy bands in k-space
    E_upper = np.zeros(N)
    E_lower = np.zeros(N)

    for i, k in enumerate(k_vals):
        # f(k) = v + w e^{-ik}
        f = v + w * np.exp(-1j * k)
        # |f(k)| = sqrt(v^2 + w^2 + 2 v w cos k)
        E_lower[i], E_upper[i] = +np.abs(f), -np.abs(f)

    energy_gap = abs(v-w)
    return k_vals, E_upper, E_lower , energy_gap


if __name__ == "__main__":
    # Build the bands
    k_vals, E_upper, E_lower, energy_gap = ssh_hamiltonian_in_k(N, v, w)

    print(f"The energy gap is {energy_gap}")
    # Plotting both bands versus k
    plt.figure(figsize=(6, 4))
    plt.plot(k_vals, E_upper,  marker='.', linestyle='-', label=r'$E_{+}(k)$')
    plt.plot(k_vals, E_lower, marker='.', linestyle='-', label=r'$E_{-}(k)$')

    plt.xlabel(r'momentum $k$')
    plt.ylabel(r'Energy $E$')
    plt.title(f'SSH Dispersion (OBC) for N={N}, v={v}, w={w}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

