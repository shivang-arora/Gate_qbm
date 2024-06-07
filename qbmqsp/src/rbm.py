
from math import log

import torch
from numpy import ndarray

from src.model import MODEL
from src.utils import *


class RBM(MODEL):

    def __init__(self, data, n_hidden_nodes, k, seed, epochs=2, trained=False, momentum_coefficient=0.5, weight_decay=0.0001, quantile=0.95):
        """
        RBM constructor. Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        Args:
            data:                   Dataset to train on
            n_hidden_nodes:             Number of hidden units
            k:                      Number of Gibbs steps to do in CD-k
            seed:                   Seed to ensure same wight training
            epochs:                 Number of training iterations
            trained:                Use precomputed wights
            momentum_coefficient:   Updates wights and biases in each step of CD-k
            weight_decay:           L2-norm Regularizations
            quantile:               Used for outlier threshhold. Everything above quantile is outlier
        """
        super().__init__(n_hidden_nodes, seed, epochs, trained, quantile)
        torch.manual_seed(seed)
        self.k = k
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay

        encoded_data, bits_input_vector, num_features = self.binary_encode_data(
            data, use_folding=True)
        self.encoded_data = torch.from_numpy(encoded_data)

        self.num_visible = bits_input_vector * num_features

        self.weights_visible_to_hidden = None
        if self.trained:
            self.weights_visible_to_hidden = torch.from_numpy(dict_to_matrix(init_weights_from_saved_model(
                model=self, values=read_from_csv("weights", seed))[1]))
        else:
            self.weights_visible_to_hidden = torch.from_numpy(np.random.uniform(-1, 1, (self.num_visible, n_hidden_nodes)).astype(np.float32))
        
        self.biases_visible = torch.ones(self.num_visible) * -log((1 / self.num_visible) / (1 - (1 / self.num_visible)))
        self.biases_hidden = torch.from_numpy(np.random.uniform(-1, 1, (n_hidden_nodes)).astype(np.float32))

        self.weights_visible_to_hidden_momentum = torch.zeros(self.num_visible, n_hidden_nodes)
        self.biases_visible_momentum = torch.zeros(self.num_visible)
        self.biases_hidden_momentum = torch.zeros(n_hidden_nodes)

        self.weight_objects = [self.weights_visible_to_hidden, self.biases_visible, self.biases_hidden, self.weights_visible_to_hidden, self.weights_visible_to_hidden_momentum, self.biases_visible_momentum, self.biases_hidden_momentum]

    def sample_prob(self, prob):
        """Sample with given probability"""
        return torch.nn.functional.relu(torch.sign(prob - (torch.rand(prob.size()))))

    def propup(self, vis):
        '''Propagate the visible units activation upwards to the hidden units'''
        return torch.sigmoid(torch.matmul(vis, self.weights_visible_to_hidden) + self.biases_hidden)

    def propdown(self, hid):
        '''Propagate the hidden units activation downwards to the visible units '''
        return torch.sigmoid(torch.matmul(hid, self.weights_visible_to_hidden.t()) + self.biases_visible)

    def sample_hidden_from_visible(self, visible_sample):
        '''Compute state of hidden units given visible units '''
        hidden_mean = self.propup(visible_sample)
        hidden_sample = self.sample_prob(hidden_mean)
        return (hidden_mean, hidden_sample)

    def sample_visible_from_hidden(self, hidden_sample):
        '''Compute state of visible units given hidden units '''
        visible_mean = self.propdown(hidden_sample)
        visible_sample = self.sample_prob(visible_mean)
        return (visible_mean, visible_sample)

    def gibbs_hvh(self, hidden_sample_0):
        '''Gibbs sampling, starting from the hidden state'''
        visible_mean_1, visible_sample_1 = self.sample_visible_from_hidden(
            hidden_sample_0)
        hidden_mean_1, hidden_sample_1 = self.sample_hidden_from_visible(
            visible_sample_1)
        return (visible_mean_1, visible_sample_1, hidden_mean_1, hidden_sample_1)

    def gibbs_vhv(self, visible_sample_0):
        '''Gibbs sampling, starting from the visible state'''
        hidden_mean_1, hidden_sample_1 = self.sample_hidden_from_visible(
            visible_sample_0)
        visible_mean_1, visible_sample_1 = self.sample_visible_from_hidden(
            hidden_sample_1)
        return (hidden_mean_1, hidden_sample_1, visible_mean_1, visible_sample_1)

    def train_for_one_iteration(self, input_data, learning_rate):
        """Implement one step of CD-k

        Returns the cost and the updates weights and biases.
        """
        # Positive phase
        positive_hidden_probabilities, positive_hidden_samples = self.sample_hidden_from_visible(
            input_data)
        positive_associations = torch.matmul(
            input_data.t(), positive_hidden_samples)

        # Negative phase
        hidden_activations = positive_hidden_samples

        for _ in range(self.k):
            nvisible_mean, nvisible_sample, nhidden_mean, nhidden_sample = self.gibbs_hvh(
                hidden_activations)
            hidden_activations = nhidden_mean

        negative_visible_sample = nvisible_sample
        negative_hidden_sample = nhidden_sample
        negative_visible_probabilities = nvisible_mean
        negative_hidden_probabilities = nhidden_mean

        negative_associations = torch.matmul(
            negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_visible_to_hidden_momentum *= self.momentum_coefficient
        self.weights_visible_to_hidden_momentum += (positive_associations -
                                                    negative_associations)

        self.biases_visible_momentum *= self.momentum_coefficient
        self.biases_visible_momentum += torch.sum(
            input_data - negative_visible_sample, dim=0)
        #self.biases_visible_momentum += torch.mean(input_data - negative_visible_sample, dim=0)

        self.biases_hidden_momentum *= self.momentum_coefficient
        self.biases_hidden_momentum += torch.sum(
            positive_hidden_probabilities - negative_hidden_probabilities, dim=0)
        #self.biases_hidden_momentum += torch.mean(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights_visible_to_hidden += self.weights_visible_to_hidden_momentum * learning_rate / batch_size
        self.biases_visible += self.biases_visible_momentum * learning_rate / batch_size
        self.biases_hidden += self.biases_hidden_momentum * learning_rate / batch_size

        self.weights_visible_to_hidden -= self.weights_visible_to_hidden * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.mean(
            (input_data - negative_visible_sample), dim=0)
        error = error.numpy()

        return error

    def train_model(self, batch_size=10, learning_rate=0.005):
        """
        Train the model with the given Parameters
        """

        batches_data = self.encoded_data.split(batch_size)
        
        self.error_container = Errcol(self.epochs, self.num_visible * len(batches_data))

        for epoch in range(self.epochs):
            epoch_error = np.array([])
            print(f'Epoch {epoch+1}')
            for batch in batches_data:
                batch = batch.view(len(batch), self.num_visible)  # flatten input data

                batch_error = self.train_for_one_iteration(batch, learning_rate)
                epoch_error = np.concatenate((epoch_error, batch_error))

            self.error_container.add_error(np.copy(epoch_error))

        self.error_container.plot("rbm_errors.pdf")

        self.calculate_outlier_threshold(self.quantile)

    def free_energy(self, visible_sample):
        '''Function to compute the free energy '''
        if type(visible_sample) == ndarray:
            visible_sample = torch.from_numpy(visible_sample)
        hidden_activations = torch.matmul(
            visible_sample, self.weights_visible_to_hidden) + self.biases_hidden
        visible_bias_term = torch.matmul(visible_sample, self.biases_visible)
        hidden_term = torch.sum(torch.log(1 + torch.exp(hidden_activations)))

        return -hidden_term - visible_bias_term

    def calculate_outlier_threshold(self, quantile=0.95):
        self.quantile = quantile
        energies = []
        for datapoint in self.encoded_data:
            energy = self.free_energy(datapoint)
            energies.append(energy)
        energies = np.array(energies)
        self.outlier_threshold = np.quantile(energies, quantile)
