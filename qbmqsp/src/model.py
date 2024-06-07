import abc
from pathlib import Path, PurePath

import numpy as np
from matplotlib import pyplot as plt

from qbmqsp.src.utils import Errcol


class MODEL(metaclass=abc.ABCMeta):

    # TODO: add num_visible here and rename dim_input in qbm
    def __init__(self, n_hidden_nodes, seed, epochs, trained, quantile) -> None:
        np.random.seed(seed)

        self.n_hidden_nodes: int = n_hidden_nodes
        self.num_visible : int = 0
        self.seed: int = seed
        self.epochs: int = epochs
        self.trained: bool = trained
        self.quantile: float = quantile

        self.error_container: Errcol = None
        self.encoded_data = None
        self.weights_visible_to_hidden = None
        self.biases_visible = None
        self.biases_hidden = None
        self.outlier_threshold: float = 0

    @abc.abstractmethod
    def free_energy(self, visible_sample):
        """Function to compute the free energy """
        return

    @abc.abstractmethod
    def calculate_outlier_threshold(self, quantile):
        """
        Function to compute the outlier threshold
        based on the quantile of the free energy
        """

    @abc.abstractmethod
    def train_model(self, batch_size, learning_rate):
        """
        Train the model with the given Parameters
        """

    @abc.abstractmethod
    def train_for_one_iteration(self, batch, learning_rate):
        """
        Compute one training iteration with the given Parameters
        """

    def predict_point_as_outlier(self, input):
        energy = self.free_energy(input)
        if energy >= self.outlier_threshold:
            return 1, energy
        return 0, energy

    @staticmethod
    def binary_encode_data(data, use_folding=False):
        """ Encode a numpy array of form [[numpy.int64 numpy.int64] ...] into a
        list of form [[int, int, int, ...], ...].
        Example: encode [[107  73] [113  90] ...] to
        [[1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],[1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0] .
        """

        # find out how many bits we need for each feature
        number_bits = len(np.binary_repr(np.amax(data)))
        number_features = data.shape[1]

        binary_encoded = ((data.reshape(-1, 1) & np.array(2 **
                          np.arange(number_bits-1, -1, -1))) != 0).astype(np.float32)
        if use_folding:
            return binary_encoded.reshape(len(data), number_features*number_bits), number_bits, number_features
        else:
            return binary_encoded.reshape(len(data), number_features, number_bits), number_bits, number_features

    @staticmethod
    def get_binary_outliers(dataset, outlier_index: int):
        outliers = np.array([entry[:-1]
                            for entry in dataset if entry[-1] >= outlier_index])

        return MODEL.binary_encode_data(outliers, use_folding=False)[0]

    @staticmethod
    def get_binary_cluster_points(dataset, cluster_index: int) -> np.ndarray:
        points = np.array([entry[:-1]
                           for entry in dataset if entry[-1] <= cluster_index])

        return MODEL.binary_encode_data(points, use_folding=False)[0]

    @staticmethod
    def plot_energy_diff(data, outlier_threshold, fname):
        # Actual plotting
        # Display the plot
        fig = plt.figure(0)
        fig.suptitle('Point Energy', fontsize=14, fontweight='bold')

        ax = fig.add_subplot()
        ax.boxplot(data, showfliers=False, showmeans=True)
        ax.set_xticklabels(['outlier', 'cluster points'], fontsize=8)

        ax.set_ylabel('Energy')

        plt.axhline(outlier_threshold)

        plt.plot([], [], '-', linewidth=1, color='orange', label='median')
        plt.plot([], [], '^', linewidth=1, color='green', label='mean')
        plt.legend()

        plt.savefig(fname)

    @staticmethod
    def plot_energy_diff_multiple(data, keys, ymin, ymax, outlier_thresholds, figure, fname):
        # Actual plotting
        # Display the plot
        thresholds = list(outlier_thresholds)
        fig = plt.figure(figure)
        fig.suptitle('Point Energy', fontsize=14, fontweight='bold')

        ax = fig.add_subplot()

        xlabels = []
        i = 0
        for _ in keys:
            xlabels.append("cluster")
            xlabels.append("outlier")
            i += 2

        ax.set_xticklabels(xlabels, fontsize=8)

        raw_data = []
        index = 0
        box_index = 0
        for key, (cluster, outlier) in data.items():
            if key in keys:
                mean = np.linalg.norm(np.concatenate([cluster,outlier]))
                raw_data.append(cluster*10/mean)
                raw_data.append(outlier*10/mean)
                thresholds[index] = thresholds[index]*10/mean
                xmin = (box_index)/int(len(keys)*2)
                xmax = (box_index+2)/int(len(keys)*2)
                ax.axhline(y=thresholds[index], xmin=xmin, xmax=xmax)
                box_index += 2
            index += 1

        box = ax.boxplot(raw_data, showfliers=False, showmeans=True, vert=True, patch_artist=True)

        boxes = []
        i = 0
        for _ in keys:
            boxes.append(box["boxes"][i])
            i += 2

        colors = ['pink', 'lightblue', 'lightgreen', '#faed4e']
        colors = colors[:len(keys)]
        colors = [ele for ele in colors for i in range(2)]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        ax.legend(boxes, keys, loc='upper right')

        ax.set_ylabel('Energy')

        plt.ylim(ymin, ymax)
        plt.savefig(fname)

    @staticmethod
    def plot_hist(data_points, data_outlier, outlier_threshold, fname):
        n_bins = 10

        fig, ax= plt.subplots()
        ax.hist([data_points, data_outlier], n_bins, histtype='bar', label=['Cluster', 'Outlier'], log=True)
        ax.set_title('Energy Histogram')

        plt.axvline(outlier_threshold, label='Threshhold',color ='black')
        plt.legend()

        plt.savefig(fname)
