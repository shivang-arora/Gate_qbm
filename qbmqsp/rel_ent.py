"""Quantum relative entropy"""
import pennylane as qml
from pennylane import numpy as np
from pennylane.math.utils import allclose, is_abstract, cast, cast_like


def _check_density_matrix(density_matrix):
    """Check the shape, the trace and the positive semi-definitiveness of a matrix."""
    dim = density_matrix.shape[-1]
    if (
        len(density_matrix.shape) not in (2, 3)
        or density_matrix.shape[-2] != dim
        or not np.log2(dim).is_integer()
    ):
        raise ValueError("Density matrix must be of shape (2**N, 2**N) or (batch_dim, 2**N, 2**N).")

    if len(density_matrix.shape) == 2:
        density_matrix = qml.math.stack([density_matrix])

    if not is_abstract(density_matrix):
        for dm in density_matrix:
            # Check trace
            trace = np.trace(dm)
            if not allclose(trace, 1.0, atol=1e-10):
                raise ValueError("The trace of the density matrix should be one.")

            # Check if the matrix is Hermitian
            conj_trans = np.transpose(np.conj(dm))
            if not allclose(dm, conj_trans):
                raise ValueError("The matrix is not Hermitian.")

            # Check if positive semi-definite
            evs, _ = qml.math.linalg.eigh(dm)
            evs = np.real(evs)
            evs_non_negative = [ev for ev in evs if ev >= -1e-7]
            if len(evs) != len(evs_non_negative):
                raise ValueError("The matrix is not positive semi-definite.")
            
            
def relative_entropy(state0, state1, base=None, check_state=False, c_dtype="complex128"):
    # Cast as a c_dtype array
    state0 = cast(state0, dtype=c_dtype)

    # Cannot be cast_like if jit
    if not is_abstract(state0):
        state1 = cast_like(state1, state0)

    if check_state:
        # pylint: disable=expression-not-assigned
        _check_density_matrix(state0)
        _check_density_matrix(state1)

    # Compare the number of wires on both subsystems
    if qml.math.shape(state0)[-1] != qml.math.shape(state1)[-1]:
        raise qml.QuantumFunctionError("The two states must have the same number of wires.")

    return _compute_relative_entropy(state0, state1, base=base)


def _compute_relative_entropy(rho, sigma, base=None):
    r"""
    Compute the quantum relative entropy of density matrix rho with respect to sigma.

    .. math::
        S(\rho\,\|\,\sigma)=-\text{Tr}(\rho\log\sigma)-S(\rho)=\text{Tr}(\rho\log\rho)-\text{Tr}(\rho\log\sigma)
        =\text{Tr}(\rho(\log\rho-\log\sigma))

    where :math:`S` is the von Neumann entropy.
    """
    if base:
        div_base = np.log(base)
    else:
        div_base = 1

    evs_rho, u_rho = qml.math.linalg.eigh(rho)
    evs_sig, u_sig = qml.math.linalg.eigh(sigma)

    # cast all eigenvalues to real
    evs_rho, evs_sig = np.real(evs_rho), np.real(evs_sig)
    ############ NEW #########################################
    # Set eigenvalues to zero if they are negative and close to zero due to finite machine precision
    evs_rho[np.isclose(evs_rho, 0).numpy() * evs_rho < 0] = 0
    evs_sig[np.isclose(evs_sig, 0).numpy() * evs_sig < 0] = 0
    ##########################################################

    # zero eigenvalues need to be treated very carefully here
    # we use the convention that 0 * log(0) = 0
    evs_sig = qml.math.where(evs_sig == 0, 0.0, evs_sig)
    rho_nonzero_mask = qml.math.where(evs_rho == 0.0, False, True)

    ent = qml.math.entr(qml.math.where(rho_nonzero_mask, evs_rho, 1.0))

    # whether the inputs are batched
    rho_batched = len(qml.math.shape(rho)) > 2
    sig_batched = len(qml.math.shape(sigma)) > 2

    indices_rho = "abc" if rho_batched else "bc"
    indices_sig = "abd" if sig_batched else "bd"
    target = "acd" if rho_batched or sig_batched else "cd"

    # the matrix of inner products between eigenvectors of rho and eigenvectors
    # of sigma; this is a doubly stochastic matrix
    rel = qml.math.einsum(
        f"{indices_rho},{indices_sig}->{target}", np.conj(u_rho), u_sig, optimize="greedy"
    )
    rel = np.abs(rel) ** 2

    if sig_batched:
        evs_sig = qml.math.expand_dims(evs_sig, 1)

    rel = qml.math.sum(qml.math.where(rel == 0.0, 0.0, np.log(evs_sig) * rel), -1)
    rel = -qml.math.sum(qml.math.where(rho_nonzero_mask, evs_rho * rel, 0.0), -1)

    return (rel - ent) / div_base