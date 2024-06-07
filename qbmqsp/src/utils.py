import csv
from ast import literal_eval
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path, PurePath


class Errcol(object):
    """Collects errors of visible neurons (aka reconstruction errors) and works with them"""

    # Absprache: num_single_errors = num visible neurons * num batches
    def __init__(self, epochs: int, num_single_errors: int):
        super(Errcol, self).__init__()
        self.errors = np.zeros((epochs, num_single_errors))
        self.epoch_counter = 0

    def add_error(self, err: list):
        '''Adds a list of errors to the epoch array
        '''
        self.errors[self.epoch_counter] = np.array(err)
        self.epoch_counter += 1

    def plot(self, fname):
        # process raw errors
        data = np.absolute(self.errors)
        save_output(title=fname, object=data)

        # Actual plotting
        plt.boxplot(data.T, showfliers=False, showmeans=True)
        plt.xlabel('epochs')
        plt.ylabel('error')
        # plt.show()
        plt.plot([], [], '-', linewidth=1, color='orange', label='median')
        plt.plot([], [], '^', linewidth=1, color='green', label='mean')
        plt.legend()
        plt.savefig(fname + ".pdf")


def import_dataset(path: str) -> np.ndarray:
    data = np.load(path)
    return data


def split_dataset_labels(dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = dataset[:, :-1]
    labels = dataset[:, [-1]].flatten()

    return data, labels

#TODO: adjust


def init_weights_from_saved_model(model, values):
    # visible nodes
    Q_vh = dict()
    # hidden layers
    Q_hh = dict()
    # "numbering of nodes"
    # order in rising numbers: visible nodes, hidden nodes

    for (connection, weight) in values:
        # first neuron of connection is visible neuron
        if connection[0] < model.num_visible:
            Q_vh[connection] = weight
        else:  # both neurons of connection are hidden neurons
            Q_hh[connection] = weight
    return Q_hh, Q_vh


# to write accuracies or initial weights of certain hyperparameter setting with different seeds to file
def write_to_csv(filename: str, content_type: str, content: list, seed: float):
    with open(filename, 'a+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        seed_with_title = ['Seed:']
        seed_with_title.append(seed)
        filewriter.writerow(seed_with_title)
        content_line = [content_type + ': ']
        content_line.extend(content)
        filewriter.writerow(content_line)
        filewriter.writerow([])


def read_from_csv(filename: str, seed: float):
    with open(filename, 'r') as csvfile:
        file = csv.reader(csvfile, delimiter=',',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
        values = None
        use_next_row = False
        for row in file:
            if len(row) >= 1:  # line not empty
                if row[0] == 'Seed:' and row[1] == str(seed):
                    use_next_row = True
                elif use_next_row:
                    values = row[1:]
                    break
        return list(map(literal_eval, values))


def dict_to_matrix(dict: Dict[Tuple[int, int], float]) -> np.ndarray:
    keys = list(dict.keys())
    if len(keys[0]) != 2:
        raise Exception("Keys of dict must have shape (int,int)")
    shape = tuple(x+1 for x in keys[-1])
    new_matrix = np.zeros(shape, dtype='float32')
    for (i, j), v in dict.items():
        new_matrix[i][j] = v

    return new_matrix


def save_output(title, object):
    path = PurePath()
    path = Path(path / 'output')
    path.mkdir(mode=0o770, exist_ok=True)
    np.save(path / title, object, allow_pickle=False, fix_imports=False)


def split_data(data: np.ndarray, clusters: int):
    labels = sorted(set(data[:, -1]))
    clusterlabels = labels[:clusters]

    training = []
    test = []
    for label in clusterlabels:
        clusterpoints = data[data[:, -1] == label]
        rows = clusterpoints.shape[0]
        if rows % 2 == 0:
            training.append(clusterpoints[:rows//2, :])
            test.append(clusterpoints[rows//2:, :])
        else:
            training.append(clusterpoints[:(rows+1)//2, :])
            test.append(clusterpoints[(rows+1)//2:, :])

    outliers = data[data[:, -1] > max(clusterlabels)]
    rows = outliers.shape[0]
    if rows % 2 == 0:
        training.append(outliers[:rows//2, :])
        test.append(outliers[rows//2:, :])
    else:
        training.append(outliers[:(rows+1)//2, :])
        test.append(outliers[(rows+1)//2:, :])

    training_data = np.vstack(training)
    test_data = np.vstack(test)

    np.random.shuffle(training_data)
    np.random.shuffle(test_data)

    return training_data, test_data
