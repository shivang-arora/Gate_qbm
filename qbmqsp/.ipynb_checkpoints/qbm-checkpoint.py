"""Quantum Boltzmann machine based on quantum signal processing"""
import scipy.linalg as spl

import pennylane as qml
from pennylane import numpy as np
from pennylane.pauli.utils import string_to_pauli_word

from .hamiltonian import Hamiltonian
from .qsp_phase_engine import QSPPhaseEngine
from .qevt import QEVT
from .rel_ent import relative_entropy


class QBM(object):
    """Quantum Boltzmann machine (QBM) based on quantum signal processing.

    Parameters
    ----------
    β, enc:
        Same as attributes.
    h, θ:
        See qbmqsp.hamiltonian.Hamiltonian
    δ, polydeg:
        See qbmqsp.qsp_phase_engine.QSPPhaseEngine

    Attributes
    ----------
    β : float
        Inverse temperature.
    enc : str in {'general', 'lcu'}
        Unitary block encoding scheme.
    H : qbmqsp.hamiltonian.Hamiltonian
        Constructed from parameters (h, θ).
    qsp : qbmqsp.qsp_phase_engine.QSPPhaseEngine
        Constructed from parameters (δ, polydeg).
    qevt : qbmqsp.qevt.QEVT
        Quantum eigenvalue transform to realize matrix function f(A) = exp(- τ * |A|). Updated after each training epoch.
    observables : qml.operation.Observable
        Observables w.r.t which the QBM is measured to optimize via gradient descent.
    aux_wire, enc_wires, sys_wires, env_wires : list[int]
        Quantum register wires of quantum circuit that prepares and measures the QBM.
    """
    
    def __init__(self, h: list[str], θ: np.ndarray[float], enc: str, δ: float, polydeg: int, β: float) -> None:
        if β < 0:
            raise ValueError("__init__: β must not be negative.")
        self.β = β
        self.enc = enc
        self.H = Hamiltonian(h, θ)
        self.qsp = QSPPhaseEngine(δ, polydeg)
        self.qevt = self._construct_qevt()
        self.aux_wire, self.enc_wires, self.sys_wires, self.env_wires = self._construct_wires()
        self.observables = self._construct_obervables()
    
    def n_qubits(self, registers: str | set[str] = None) -> int:
        """Return number of qubits per registers.
        
        Parameters
        ----------
        registers : str | set[str]
            Quantum registers whose number of qubits should be returned.
            Must be an element from or a subset of {'aux', 'enc', 'sys', 'env'}.

        Returns
        -------
        n : int
            Number of qubits used per registers.
        """
        if registers is None:
            registers = {'aux', 'enc', 'sys', 'env'}
        elif type(registers) == str:
            registers = {registers}
        if not registers.issubset({'aux', 'enc', 'sys', 'env'}):
            raise ValueError("n_qubits: registers must be an element from or a subset of %r." % {'aux', 'enc', 'sys', 'env'})
        
        n = 0
        if 'env' in registers:
            n += self.qevt.n_qubits('sys')
        registers.discard('env')
        if len(registers) != 0:
            n += self.qevt.n_qubits(registers)
        return n

    def _generate_qsp_phases(self) -> np.ndarray[float]:
        τ = self.β / (1-self.qsp.δ) * self.H.θ_norm()
        φ = self.qsp.generate(τ)
        return φ

    def _construct_qevt(self) -> QEVT:
        φ = self._generate_qsp_phases()
        h_δ, θ_δ = self.H.preprocessing(self.qsp.δ)
        return QEVT(h_δ, θ_δ, self.enc, φ)
    
    def _construct_wires(self) -> tuple[list[int], list[int], list[int], list[int]]:
        wires = list(range(self.n_qubits()))
        aux_wire = wires[: self.n_qubits('aux')]
        enc_wires = wires[self.n_qubits('aux') : self.n_qubits({'aux', 'enc'})]
        sys_wires = wires[self.n_qubits({'aux', 'enc'}) : self.n_qubits({'aux', 'enc', 'sys'})]
        env_wires = wires[self.n_qubits({'aux', 'enc', 'sys'}) : self.n_qubits({'aux', 'enc', 'sys', 'env'})]
        return aux_wire, enc_wires, sys_wires, env_wires

    def _construct_obervables(self) -> list[qml.operation.Observable]:
        n_aux_enc = self.n_qubits({'aux', 'enc'})
        aux_enc_wires = self.aux_wire + self.enc_wires
        proj0 = qml.Projector( [0] * n_aux_enc, aux_enc_wires)

        new_sys_wires = list(range(self.n_qubits('sys')))
        wire_map = dict(zip(self.sys_wires, new_sys_wires))
        observables = [proj0] + [proj0 @ string_to_pauli_word(self.H.h[i], wire_map) for i in range(self.H.n_params)]
        return observables
    
    def _bell_circuit(self) -> None:
        for i, j in zip(self.sys_wires, self.env_wires):
            qml.Hadamard(i)
            qml.CNOT([i, j])

    def _prepare(self) -> None:
        self._bell_circuit()
        self.qevt.circuit(self.aux_wire, self.enc_wires, self.sys_wires)
    
    def _measure(self) -> None:
        return [qml.expval(self.observables[i]) for i in range(len(self.observables))]
    
    def _compute_expvals(self) -> np.ndarray[float]:
        dev = qml.device('default.qubit', wires=self.n_qubits())
        @qml.qnode(dev)
        def quantum_circuit():
            self._prepare()
            return self._measure()
        
        measurements = quantum_circuit()
        success_probabilty = measurements[0]
        qbm_expvals = measurements[1:] / success_probabilty
        return qbm_expvals
    
    def _loss_func(self, ρ0: np.ndarray[float], ρ1: np.ndarray[float]) -> float:
        return relative_entropy(ρ0, ρ1, check_state=True).item()
    
    def assemble(self) -> np.ndarray[float]:
        """Assemble QBM."""
        expH = spl.expm(-self.β * self.H.assemble())
        return expH / np.trace(expH)
    
    def train(self, ρ_data: np.ndarray[float], learning_rate: float, epochs: int) -> tuple[list[float], list[float]]:
        """Train QBM to fit the optimal model for ρ_data in terms of Jaynes' principle.

        To that end, gradient descent is perfomed using the quantum relative entropy as a loss function.
        
        Parameters
        ----------
        ρ_data : np.ndarray[dtype=float, ndim=2]
            Density matrix encoding the target probabilty distribution.
        learning_rate : float
            Learning rate of gradient descent update.
        epochs : int
            Number of training epochs.
        
        Returns
        -------
        losses : list[float]
            List containing the loss after each epoch.
        aa_grad_θs : list[float]
            List containing the absolute average of the gradient of the loss function w.r.t. θ after each epoch
            This is equivalent to the average absolute deviation of the expectation values between QBM and ρ_data in terms of Jaynes' principle.
        """
        if not ρ_data.ndim == 2:
            raise ValueError("train: ρ_data must be a 2D array.")

        ρ_data_expvals = np.array([np.trace(ρ_data @ self.H.assemble(i)) for i in range(self.H.n_params)])

        losses, aa_grad_θs = [], []
        for epoch in range(1, epochs + 1):
                
                qbm_expvals = self._compute_expvals()
                grad_θ = ρ_data_expvals - qbm_expvals
                self.H.θ = self.H.θ - learning_rate * grad_θ
                
                self.qevt = self._construct_qevt()
                
                qbm = self.assemble()
                loss = self._loss_func(ρ_data, qbm)
                losses.append(loss), aa_grad_θs.append(np.mean(np.abs(grad_θ)).item())

                if epoch % 10 == 0:
                        print("Epoch %d: relative entropy S = %r" % (epoch, loss))
                        print("          absolute average of grad_θ = %r" % aa_grad_θs[-1])
        
        print("\nFinal relative entropy S = %r" % loss)
        print("Final absolute average of grad_θ = %r" % aa_grad_θs[-1])
        print("Final quadratic error = %r" % np.sum((qbm - ρ_data)**2))

        return losses, aa_grad_θs