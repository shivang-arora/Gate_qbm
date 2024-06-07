#!/usr/bin/env python
import argparse
from pathlib import Path, PurePath

import numpy as np
from sklearn.datasets import make_blobs

# TODO: Add Variance to file name
parser = argparse.ArgumentParser(
    description='Generate clustered datasets with outliers.')
parser.add_argument('-d', '--dimensions',
                    metavar='INT',
                    help='Amount of dimensions per datapoint',
                    default=2,
                    type=int)
parser.add_argument('-c', '--clusters',
                    metavar='INT',
                    help='Amount of clusters',
                    default=3,
                    type=int)
parser.add_argument('-p', '--points',
                    metavar='INT',
                    help='Amount of datapoints per cluster',
                    default=300,
                    type=int)
parser.add_argument('-r', '--range',
                    metavar='INT',
                    help='Value range for all dimensions',
                    default=[1, 127],
                    nargs=2,
                    type=int)
parser.add_argument('-v', '--variance',
                    metavar='FLOAT',
                    help="Clusters' variance",
                    default=1.0,
                    type=float)
parser.add_argument('-o', '--outliers',
                    metavar='INT',
                    help='Amount of outliers',
                    default=1,
                    type=int)
parser.add_argument('-b', '--batch',
                    metavar='INT',
                    help='Amount of generated datasets',
                    default=1,
                    type=int)
parser.add_argument('-l', '--labels',
                    help='Additionally stores labels if present',
                    action='store_true')


flags = parser.parse_args()


def create_blobset(flags: argparse.Namespace) -> np.ndarray:
    ''' Generates a Dataset of type numpy.ndarray.
        All Datapoints are vectors of integers.
        Consists of Clusters and a few outliers depending on user specification.
        Contains int(clusters*points+outliers) many points.
    '''
    dims = flags.dimensions
    clusters = flags.clusters
    points = flags.points
    valrange = tuple(flags.range)
    variance = flags.variance
    outliers = flags.outliers

    # Generate the definition of all clusters
    # A cluster containing only a single element is an outlier
    real_clusters = [points for i in range(clusters)]
    outliers = [1 for i in range(outliers)]
    clusters = real_clusters + outliers

    dataset_float, labels = make_blobs(n_samples=clusters, n_features=dims,
                                       centers=None, center_box=valrange, cluster_std=variance)

    # reduce floats to ints to save qubits
    # should not make a difference on dense clusters in a high value range
    dataset = dataset_float.astype(int)

    return dataset, labels


def save_dataset(dataset: np.ndarray, fname: str):
    ''' Saves numpy.ndarray in binary format.
        File can be found in folder datasets which will be created in
        the path where you called the script from.
        Paths are (hopefully) computed OS independent.
    '''
    path = PurePath()
    path = Path(path / 'datasets')
    path.mkdir(mode=0o770, exist_ok=True)
    np.save(path / fname, dataset, allow_pickle=False, fix_imports=False)


# Actual generation of datasets
for i in range(1, flags.batch+1):
    data, labels = create_blobset(flags)
    name = f'o{flags.outliers}_c{flags.clusters}_d{flags.dimensions}_p{flags.points}_{i}'
    if flags.labels:
        name = f'l_o{flags.outliers}_c{flags.clusters}_d{flags.dimensions}_p{flags.points}_{i}'
        data = np.concatenate((data, labels.reshape(len(labels), 1)), axis=1)

    save_dataset(data, name)
