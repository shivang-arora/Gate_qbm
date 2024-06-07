"""Generate data"""
import scipy.special as sps
import scipy.linalg as spl
import networkx as nx

from pennylane import numpy as np

from qbmqsp.hamiltonian import Hamiltonian
from qbmqsp.utils import int_to_bits


## Generate quantum data
def xxz_hamiltonian(n: int, J: float = 0.5, Δ: float = 0.75) -> Hamiltonian:
    """Construct the Heisenberg XXZ Hamiltonian."""
    h = list()
    for p in ['X', 'Y', 'Z']:
        for i in range(n - 1):
            pauli_string = list(n * 'I')
            pauli_string[i] = pauli_string[i+1] = p
            h.append(''.join(pauli_string))
    θ = np.array((n-1)*[J] + (n-1)*[J] + (n-1)*[Δ])
    return Hamiltonian(h, θ)

def xxz_gibbs_state(n_qubits: int, J: float = 0.5, Δ: float = 0.75, β: float = 1.) -> np.ndarray[float]:
    """Construct the Gibbs state of the Heisenberg XXZ model at inverse temperature β."""
    H = xxz_hamiltonian(n_qubits, J, Δ)
    expH = spl.expm(-β * H.assemble())
    ρ = expH / np.trace(expH)
    return ρ


## Generate classical data
def basis_encoding(p):
    """Encode discrete probability distribution into a rank-1 density matrix.

        Parameters
        ----------
        p : list[float]
            Discrete probability distribution.

        Returns
        -------
        ρ : np.ndarray
            Rank-1 density matrix |Ψ><Ψ| with |Ψ> encoding sqrt(p) in its amplitude.
    """
    p_sqrt = np.sqrt(p)
    ρ = np.outer(p_sqrt, p_sqrt)
    return ρ

# Boltzmann
def gen_boltzmann_dist(n, β=1):
    """Generate NP-hard Boltzmann distribution of a random classical Ising model."""
    seed = 42
    G = nx.fast_gnp_random_graph(n, 1/n)
    connected = nx.is_connected(G)
    while not connected:
        G = nx.fast_gnp_random_graph(n, 1/n, seed=seed)
        connected = nx.is_connected(G)
        seed += seed
    for (i, j) in G.edges:
        G.edges[i, j]['weight'] = np.random.random()
    W = nx.adjacency_matrix(G)

    Es = []
    for k in range(2**n):
        s = np.array([2*int(bit) - 1 for bit in int_to_bits(k, n)])
        E = s @ W @ s / 2
        Es.append(E)
    Es = np.array(Es)
    return np.exp(-β*Es) / np.exp(-β*Es).sum()

# Gauss
def gen_discrete_gauss_dist(n):
    """Generate Gaussian distribution."""
    f = np.array([sps.comb(n - 1, i, exact=True) for i in range(n)], dtype='O')
    f = np.float64(f)/np.float64(f).sum()

    if not np.allclose(f.sum(), 1.0):
        raise ValueError("The distribution sum is not close to 1.\n" 
                         "f.sum(): %s" % f.sum())
    return f