"""LCU Hamiltonian"""
from typing import Optional

from pennylane import numpy as np


class Hamiltonian(object):
    """Representation of a LCU Hamiltonian consisting of Pauli string operators.
        
        The Hamiltonian must have the form H = \\sum_i θ[i]*h[i], where θ[i] is the real coefficient of the i-th Pauli string h[i].
        h[i][j] represents a Pauli operator acting on the j-th qubit.
        
        Example: 
        The Hamiltonian of the TFI model H = - J \\sum_{i} \\sigma^x_i \\sigma^x_{i+1} - g \\sum_{i} \\sigma^z_i acting on 3 qubits could be represented by
        θ = [-J, -J, -g, -g, -g]
        h = ['XXI', 'IXX', 'ZII', 'IZI', 'IIZ']

    Parameters
    ----------
    h, θ:
        Same as attributes.

    Attributes
    ----------
    h : list[str]
        List of Pauli string operator representations, where h[i][j] is from {'I', 'X', 'Y', 'Z'}.
    θ : np.ndarray[dtype=float, ndim=1]
        Hamiltonian parameters. θ[i] is the coefficient of the i-th Pauli string h[i].
    n_qubits : int
        Number of qubits the Hamiltonian acts on.
    n_params : int
        Number of Hamiltonian parameters.
    """
    
    def __init__(self, h: list[str], θ: np.ndarray[float]) -> None:
        self.θ = np.asarray(θ)
        if self.θ.ndim != 1:
            raise ValueError("__init__: θ must have dimension 1.")
        self.h = h
        self.n_qubits = len(h[0])
        self.n_params = len(h)
        if self.n_params != len(self.θ):
            raise ValueError("__init__: h and θ must have same length.")
        
    def θ_norm(self) -> float:
        """Compute L1-norm of θ."""
        return np.linalg.norm(self.θ, ord=1)
    
    def preprocessing(self, δ: float) -> tuple[list[str], np.ndarray[float]]:
        """Preprocess Hamiltonian to scale its spectrum to the interval [δ, 1]."""
        if not δ < 1:
            raise ValueError("preprocessing: δ must be smaller than 1.")
        
        h_δ = self.h + [self.n_qubits * 'I']
        θ_δ = np.append(self.θ * (1-δ)/(2*self.θ_norm()), (1+δ)/2)
        return h_δ, θ_δ
    
    def assemble(self, i: Optional[int] = None) -> np.ndarray[float]:
        """Assemble pauli string operator h[i] or full Hamiltonian."""
        def assemble_pauli_string(pauli_string: str):
            σ = {'I': np.array([[1., 0.], [0., 1.]]), 
                 'X': np.array([[0., 1.], [1., 0.]]), 
                 'Y': np.array([[0., -1j], [1j, 0.]]), 
                 'Z': np.array([[1., 0.], [0., -1.]])}
            pauli_string_operator = 1
            for p in pauli_string[::-1]:
                pauli_string_operator = np.kron(σ[p], pauli_string_operator)
            return pauli_string_operator
        
        if i is None:
            H_operator = 0
            for i in range(self.n_params):
                H_operator = H_operator + self.θ[i] * assemble_pauli_string(self.h[i])
            return H_operator
        else:
            if not 0 <= i < self.n_params:
                raise ValueError("assemble: i must be from {0,..,%r}." % (self.n_params))
            return assemble_pauli_string(self.h[i])