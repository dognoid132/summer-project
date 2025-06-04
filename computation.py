import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def ssh_hamiltonian(N: int, v: float, w: float):
    """
    The real-space SSH Hamiltonian with open boundary conditions (C_{N+1}=0).

    Parameters:
    N : Number of unit cells (total sites = 2*N).
    v, w : Hopping amplitudes

    Returns:
    H : The SSH Hamiltonian matrix in real space basis.
    """
    dim = 2 * N
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    # Construct the matrix element <m,alpha|H|n,beta>
    for i in range(N):
        index_A = 2 * i
        index_B = 2 * i + 1

        # Hopping term v between (i, A) and (i, B)
        hamiltonian[index_A, index_B] = v
        hamiltonian[index_B, index_A] = np.conjugate(v)

        # Hopping w between (i, B) and (i+1, A), if i < N-1
        if i < N - 1:
            hamiltonian[index_B, index_A + 2] = w
            hamiltonian[index_A + 2, index_B] = np.conjugate(w)

    return hamiltonian

if __name__ == "__main__":
    # setting up constants
    N = 500
    v = 1.0
    w = 0.5

    # results
    H_ssh = ssh_hamiltonian(N, v, w)
    eigvals, eigvecs = eigh(H_ssh)


    # Plotting eigenvalues in ascending order
    plt.figure()
    plt.plot(np.arange(len(eigvals)), eigvals, marker='o', linestyle='-')
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigen-energy")
    plt.title(f"SSH eigenvalues (OBC) for N={N} cells")
    plt.grid(True)
    plt.tight_layout()
    plt.show()