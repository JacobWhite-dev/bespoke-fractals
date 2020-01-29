import math
import numpy as np
import pandas as pd
import sklearn.manifold
import metrics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualiser():

    def __init__(self, data, labels, reducer):

        self._data = np.array(data)

        # Check if labels are dataframe
        if not isinstance(labels, pd.DataFrame):
            # Convert to dataframe with default titles
            print("Labels have not been provided as a pandas DataFrame",
                  "and will be cast to one. Default label names will be",
                  "generated and errors may occur")
            labels = pd.DataFrame(data = labels)

        self._labels = labels

        self._reducer = reducer
        self._result = None

    def set_data(self, data):
        self._data = data

    def get_data(self):
        return self._data

    def set_labels(self, labels):
        self._labels = labels

    def get_labels(self):
        return self._labels

    def set_reducer(self, reducer):
        # Should work with an sklearn pipeline
        self._reducer = reducer

    def get_reducer(self):
        return self._reducer

    def set_result(self, result):
        self._result = result

    def get_result(self):
        return self._result

    def fit(self):
        # Check all specified
        self._reducer.fit(self._data)

    def transform(self):
        # Check all specified
        self._result = self._reducer.transform(data)

    def fit_transform(self):
        # Check all specified
        self._result = self._reducer.fit_transform(self._data)

    def __plot_result_1d(self, fig, rows, cols, index):
        points = np.size(self._result)

        ax = fig.add_subplot(rows, cols, index + 1)
        ax.scatter(self._result, np.zeros((points, )), c= self._labels[index, :], cmap = 'Spectral', s = 5)

    def __plot_result_2d(self, fig, rows, cols, index):
        
        labels = self._labels.iloc[:, index]
        max_label = np.amax(labels)
        min_label = np.amin(labels)
        delta_labels = max_label - min_label + 1

        ax = fig.add_subplot(rows, cols, index + 1)
        im = ax.scatter(self._result[:, 0], self._result[:, 1], c= self._labels.iloc[:, index], cmap = 'Spectral', s = 5)
        fig.colorbar(im, boundaries = np.arange(delta_labels + 1) - 0.5, ax = ax).set_ticks(np.arange(delta_labels))
        ax.set_title(self._labels.columns[index])

    def __plot_result_3d(self, fig, rows, cols, index):

        ax = fig.add_subplot(rows, cols, index + 1, projection='3d')
        ax.scatter(self._result[:, 0], self._result[:, 1], self._result[:, 2], c = self._labels[index, :], cmap = 'Spectral', s = 5)

    def __invalid_dimensions(self, *argv, **kwargs):
        print("Data cannot be plotted.")

    def plot_result(self):
        # Check dimension of result and plot accordingly
        # Make sure to account for subplots based on dimensions of labels
        if self._result is None:
            print("No result has been calculated.")
            return

        numPlots = 1 if self._labels.ndim == 1 else self._labels.shape[1]
        rows = math.ceil(math.sqrt(numPlots))
        cols = math.ceil(numPlots / rows)

        dim = self._result.shape[1]

        plot_funcs = {1: self.__plot_result_1d, 2: self.__plot_result_2d, 3: self.__plot_result_3d}
        plot_func = plot_funcs.get(dim, self.__invalid_dimensions)

        fig = plt.figure()
        for i in range(numPlots):
            plot_func(fig, rows, cols, i)
            plt.gca().set_aspect('equal', 'datalim')
        plt.show()






        
        
       
