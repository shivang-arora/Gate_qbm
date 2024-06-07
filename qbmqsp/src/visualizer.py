#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description="""Visualizes .npy files.
                                                Handles files as well as directories.""")

parser.add_argument('-f', '--format',
                    help='One of [eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff]',
                    nargs=1,
                    default=['jpeg'],
                    type=str,
                    choices=['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps',
                             'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff'],
                    )

parser.add_argument('PATH',
                    metavar='PATH',
                    help='File or directory',
                    type=Path,
                    nargs='+')

parser.add_argument('-c', '--color',
                    help='colors outliers and clusterpoints',
                    action='store_true')

flags = parser.parse_args()


def draw_dataset(path: Path, flags: argparse.Namespace):
    ''' Will draw a dataset.
        If there are more than two dimensions a scatterplot will be drawn.
        Additionally tells you what is currently doing since plotting sometimes
        might take a while depending on the size of your dataset.
    '''
    data = np.load(path)
    dims = data.shape[1]-1

    if flags.color:
        params = path.stem.split("_")
        clusterindex = int(params[2][1:])-1

        cols = [f'{i}' for i in range(1, dims+1)]
        cols.append('label')

        # .astype({'label':str}, errors='raise')
        frame = pd.DataFrame(data, columns=cols)
        clusterpoints = frame['label'] <= clusterindex
        outliers = frame['label'] > clusterindex

        frame['label'][clusterpoints] = "cluster"
        frame['label'][outliers] = "outlier"

    else:
        data = data[:, :-1]
        frame = pd.DataFrame(data)

    if dims < 2:
        print(
            f'Dataset "{path.name}" has less than two dimensions and is therefore skipped.')
        return
    else:
        print(f'Currently drawing {path.stem}.')
        if dims == 2:
            frame.plot.scatter(0, 1, title=f'{path.name}')
        else:
            if flags.color:
                g = sns.pairplot(frame, hue="label", aspect=2)
                g.set(xticks=range(0, 127, 20), xmargin=-0.15)
            else:
                pd.plotting.scatter_matrix(frame, alpha=1, diagonal='kde')
            # plt.gcf().suptitle(f'{path.name}')

        gpath = path.parent / 'graphics'
        gpath.mkdir(mode=0o770, exist_ok=True)
        gpath /= f'{path.with_suffix("." + flags.format[0]).name}'
        plt.savefig(gpath)


def draw_dir(path: Path, flags: argparse.Namespace):
    '''Iteratively calls draw_dataset() on each file of type ".npy"
    '''
    for f in path.glob('*.npy'):
        draw_dataset(f, flags)


# Actual Drawing
for p in flags.PATH:
    if p.exists():
        if p.is_dir():
            draw_dir(p, flags)
        if p.is_file():
            draw_dataset(p, flags)
    else:
        print(f'Path "{p}" does not exist and is therefore skipped.')
