"""QEVT"""
import pennylane as qml
from pennylane import numpy as np

from .qsp import QSP
from .block_encode import BlockEncode


class QEVT(QSP):
    """Quantum eigenvalue transform implementing real matrix polynomials on a quantum computer.

    Parameters
    ----------
    h, θ, enc:
        See qbmqsp.block_encode.BlockEncode

    Attributes
    ----------
    block_encode : qbmqsp.block_encode.BlockEncode
        Constructed from parameters (h, θ, enc).
    φ : list[float]
        QSP phases.
    """

    def __init__(self, h: list[str], θ: np.ndarray[float], enc: str, φ: list[float]) -> None:
        self.block_encode = BlockEncode(h, θ, enc)
        self.φ = φ
    
    def n_qubits(self, registers: str | set[str] = None) -> int:
        """Return number of qubits.
        
        Parameters
        ----------
        registers : str | set[str]
            Quantum registers whose number of qubits should be returned.
            Must be an element from or a subset of {'aux', 'enc', 'sys'}.

        Returns
        -------
        n : int
            Total number of qubits used in registers.
        """
        if registers is None:
            registers = {'aux', 'enc', 'sys'}
        elif type(registers) == str:
            registers = {registers}
        if not registers.issubset({'aux', 'enc', 'sys'}):
            raise ValueError("n_qubits: registers must be an element from or a subset of %r." % {'aux', 'enc', 'sys'})
        
        n = 0
        for register in registers:
            if register == 'aux':
                n += 1
            elif register in {'enc', 'sys'}:
                n += self.block_encode.n_qubits(register)
        return n

    def _Π(self, φ: list[float], control: list[int], target: list[int]):
        qml.MultiControlledX(wires=control+target, control_values=[0]*len(control))
        self.S(φ, target)
        qml.MultiControlledX(wires=control+target, control_values=[0]*len(control))

    def circuit(self, aux_wire: list[int], enc_wires: list[int], sys_wires: list[int]):
        """Apply QEVT circuit.
        
        Parameters
        ----------
        aux_wire : list[int]
            Auxiliary wire used for block encoding the projector controlled phase shift operation Π_φ.
        enc_wires : list[int]
            Wires used for block encoding the Hamiltonian.
        sys_wires: list[int]
            Wires on which the Hamiltonian acts on.
        """
        if self.n_qubits('aux') != len(aux_wire):
            raise ValueError("circuit: length of aux_wire must match number of auxiliary qubits (=%r)." % self.n_qubits('aux'))
        if self.n_qubits('enc') != len(enc_wires):
            raise ValueError("circuit: length of enc_wires must match number of encoding qubits (=%r)." % self.n_qubits('enc'))
        if self.n_qubits('sys') != len(sys_wires):
            raise ValueError("circuit: length of sys_wires must match number of system qubits (=%r)." % self.n_qubits('sys'))
        
        qml.Hadamard(aux_wire)
        for k in range(1, len(self.φ)):
            qml.PauliZ(aux_wire)
            self._Π(self.φ[-k], enc_wires, aux_wire)
            self.block_encode.circuit(enc_wires, sys_wires)
        self._Π(self.φ[-len(self.φ)], enc_wires, aux_wire)
        qml.Hadamard(aux_wire)