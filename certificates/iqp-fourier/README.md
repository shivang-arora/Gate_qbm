# Fixed diagonal quantum circuits for probability distributions

This directory contains a standalone Lean 4.19.0 project with mathlib v4.19.0. The two registers begin in zero. All trainable operations are diagonal phase gates. Only the work register is measured.

| Work register | What the certificate proves |
| --- | --- |
| An n-bit string with Hadamards | One fixed circuit on n hidden and n work qubits represents **every distribution exactly** by varying its real angles. |
| An integer in 0 through d−1 with a d-point Fourier transform | With m hidden qubits the circuit represents every histogram of 2ᵐ observations exactly. For every positive tolerance, a width chosen before the target suffices for all targets. |
| A string of k base-d digits | A separate Fourier transform on each digit gives the same approximation guarantee with dᵏ possible strings. The digits can be correlated. |
| A three-level work register | The target (1/3, 2/3, 0) is impossible exactly for every finite number of hidden qubits, even when all diagonal phases are independently tunable. |

### Why the two cases differ

Each hidden address contributes one equally weighted distribution. A binary phase pattern lets one address split its probability in arbitrary proportions between any two bitstrings. Giving each address a pair of outputs and a splitting probability accounts for an arbitrary target. The cyclic Fourier transform can send an address to any one integer using a linear phase ramp. Repeating an integer at several addresses sets its probability by counting. For three levels, a general phase table that gives zero probability to the third integer can only select a single one of the other integers on each address.

### Check the certificates

With elan installed, run the following in this directory.

```sh
lake update
lake exe cache get
lake build
```

[verification.txt](verification.txt) records the axiom reports from a local Lean kernel check. The listed theorems depend only on propositional extensionality, classical choice and quotient soundness. The verification uses real-valued mathematics and ideal gates.

### Declaration index

- [The fixed Hadamard architecture and exact universality](FixedIQP.lean#L283)
- [Exact integer histograms](IntegerFourier.lean#L61)
- [Approximation of arbitrary integer distributions](IntegerFourier.lean#L132)
- [Exact obstruction for three levels](IntegerFourier.lean#L237)
- [Fourier transforms on native digit strings](IntegerFourier.lean#L290)

The hidden-address mixture construction is related to [Kurkin et al., arXiv:2510.08476](https://arxiv.org/abs/2510.08476) and their [QTML 2026 extended abstract](https://indico.nithecs.ac.za/event/12/contributions/509/).
