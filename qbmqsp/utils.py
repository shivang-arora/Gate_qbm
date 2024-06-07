"""Utility functions"""


def int_to_bits(n: int, m: int) -> list[int]:
    """Convert integer n to its m-bit binary representation."""
    return list(map(int, bin(n)[2:].zfill(m)))


def construct_fcqbm_pauli_strings(n: int) -> list[str]:
    """Construct representation of all Pauli string operators of a fully-connected n-qubit QBM.
        
        Parameters
        ----------
        n : int
            Number of qubits the QBM acts on.

        Returns
        -------
        h : list[str]
            List of Pauli string operator representations, where h[i][j] is from {'I', 'X', 'Y', 'Z'}.
    """
    h = list()
    for p in ['X', 'Y', 'Z']:

        for i in range(n):
            pauli_string = list(n * 'I')
            pauli_string[i] = p
            h.append(''.join(pauli_string))

            for j in range(i + 1, n):
                pauli_string = list(n * 'I')
                pauli_string[i] = pauli_string[j] = p
                h.append(''.join(pauli_string))
    return h

def construct_multi_fcqbm_pauli_strings(n: int) -> list[str]:
    """Same as construct_fcqbm_pauli_strings but add multi-qubit interactions."""
    h = list()
    for p in ['X', 'Y', 'Z']:

        for i in range(n):
            pauli_string = list(n * 'I')
            pauli_string[i] = p
            h.append(''.join(pauli_string))

            for j in range(i + 1, n):
                pauli_string = list(n * 'I')
                pauli_string[i] = pauli_string[j] = p
                h.append(''.join(pauli_string))

                for k in range(j + 1, n):
                    pauli_string = list(n * 'I')
                    pauli_string[i] = pauli_string[j] = pauli_string[k] = p
                    h.append(''.join(pauli_string))
    return h