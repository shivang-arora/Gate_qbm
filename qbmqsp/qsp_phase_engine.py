"""QSP phase factors for Gibbs state preparation"""
import matlab.engine

from pennylane import numpy as np


class QSPPhaseEngine(object):
    """QSP phase generator for Gibbs state preparation.

    Compute QSP phases of a polynomial approximation of f(x) = exp(- τ * |x|) on an interval [δ, 1] by interfacing with QSPPACK.
    The polynomial approximation is solved using convex optimization.

    Parameters
    ----------
    δ, polydeg:
        Same as attributes.

    Attributes
    ----------
    δ : float in [0, 1]
        Restricts interval on which polynomial approximation agrees with f(x) to [δ, 1].
    polydeg : int
        Degree of polynomial approximation of f(x).
    eng : matlab.engine.matlabengine.MatlabEngine
        Python object using MATLAB as computational engine within Python session.
    """
    
    def __init__(self, δ: float = 0.1, polydeg: int = 10) -> None:
        if not 0 <= δ <= 1:
            raise ValueError("__init__: δ must be in range %r." % [0, 1])
        
        self.δ = δ
        self.polydeg = polydeg
        print("\nStarting MATLAB engine..", end=" ", flush=True)
        
        self.eng = matlab.engine.start_matlab()
        print("Done.\n")

    def generate(self, τ: float, convention: str = 'R') -> np.ndarray[float]:
        """Compute QSP phases of a polynomial approximation of f(x) = exp(- τ * |x|).
        
        Parameters
        ----------
        τ : float
            Imaginary time parameter of f(x).
        convention : str in {'R', 'Wx'}
            QSP convention specifying which signal operator to use.

        Returns
        -------
        φ : np.ndarray[dtype=float, ndim=1]
            QSP phases.
        """
        if convention not in {'R', 'Wx'}:
            raise ValueError("generate: convention must be one of %r." % {'R', 'Wx'})
        
        φ = self.eng.phaseangles_qbm(τ, self.δ, float(self.polydeg))
        φ = np.array(φ).squeeze()
        if convention == 'R':
            φ[1:-1] -= np.pi/2
            φ[0] -= np.pi/4
            φ[-1] -= np.pi/4
        return φ
    
    def __del__(self) -> None:
        print("\nStopping MATLAB engine ...", end=" ", flush=True)
        self.eng.quit()
        print("Done.\n")