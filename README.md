# QBMQSP

The implementation of a gate model based quantum Boltzmann machine based on quantum signal processing for anamaly detection dataset.

## Description

... The current working file is in the test folder named 'Gate_qbm.ipynb'


## Getting Started

### Dependencies

...

### Installing

1. Install MATLAB release R2024a.
2. Install QSPPACK: https://github.com/qsppack/QSPPACK
3. and its dependencies: `chebfun` und `CVX`.
4. Make sure the toolbox and the file `qbmqsp/phaseangles_qbm.m` are on the MATLAB path.
5. Install the required Python environment (called `qbmqsp`) using conda by running:
```
conda env create -f env.yml
```
Next install the repository as a Python package in development mode into the previously created virtual environment.

6. First activate the environment:
```
conda activate qbmqsp
```
7. Install `conda-build` by running:
```
conda install conda-build
```
8. Finally, run `conda-develop` and specify the path to the repository and to the virtual environment:
```
conda-develop <path-to-repository> -p <env name>
```
The path to the virtual environment is shown after running `conda env list`.

### Executing program

...

## Help

...

## Authors

...

## Version History

...

## License

...

## Acknowledgments

...

## Reference

Dong, Y., Meng, X., Whaley, K.B. and Lin, L., 2021. Efficient phase-factor evaluation in quantum signal processing. Physical Review A, 103(4), p.042419.
