"""QSP theorem"""
import pennylane as qml
from pennylane import numpy as np


class QSP(object):
    """QSP gates and circuit for implementing real polynomials on a quantum computer."""
    
    def S(self, φ: float, wire: int) -> None:
        qml.RZ(-2*φ, wire)

    def W(self, x: float, wire: int) -> None:
        if abs(x) > 1:
            raise ValueError("W: x must be in range %r." % [-1, 1])
        qml.RX(-2*np.arccos(x), wire)

    def R(self, x: float, wire: int) -> None:
        qml.RZ(-np.pi/2, wire)
        self.W(x, wire)
        qml.RZ(-np.pi/2, wire)
        # qml.GlobalPhase(np.pi/2)

    def Π(self, φ: float, control: int, target: int) -> None:
        qml.CNOT([control, target])
        self.S(φ, target)
        qml.CNOT([control, target])

    def circuit(self, x: float, φ: float, wires: list[int] = [0, 1]) -> None:
        qml.Hadamard(wires[0])
        for k in range(1, len(φ)):
            qml.PauliZ(0)
            self.Π(φ[-k], wires[1], wires[0])
            self.R(x, wires[1])
        self.Π(φ[-len(φ)], wires[1], wires[0])
        qml.Hadamard(wires[0])