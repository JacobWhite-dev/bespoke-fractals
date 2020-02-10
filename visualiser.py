"""
Visualiser module for visualising latent spaces using dimensionality
reduction.

Created on Wed Jan 29 2020 

@author: uqjwhi35
"""

import math
import numpy as np
import pandas as pd
import sklearn.manifold
import metrics
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

class Visualiser():
    '''
    Class for the visualising latent spaces using dimensionality reduction.
    To do this data, labels and a reducer must be provided. 

    Data:
    Data are the points in the latent space we want to visualise. They should
    be given as a pandas DataFrame with each point occupying its own row, and
    each column representing a specific dimension in the latent space, e.g.

    ____ | dim0 | dim1 | dim2 | ...  | dimN
    pt 0 | 0.11 | 1.34 | 1.24 | ...  | 4.55
    pt 1 | 5.67 | 4.55 | 0.00 | ...  | 6.77
    ...  | ...  | ...  | ...  | ...  | ...
    pt N | 2.56 | 1.11 | 6.33 | ...  | 1.23

    The actual headers provided for each column are not important, but it
    never hurts to label your data. 

    Labels:
    Labels are what we will use to colour each point in our visualisation.
    They normally are used to distinguish between input classes, e.g. digits
    in the MNIST digits dataset. A separate sub-plot will be created for each
    type of label supplied. Labels should also be provided in the form of
    a pandas DataFrame, with each point occupying its own row, and each column
    representing the value of a given label, e.g.

    ____ | height | weight | sex  | ...  | GPA
    pt 0 | 0.11   | 1.34   | 0    | ...  | 4.55
    pt 1 | 5.67   | 4.55   | 1    | ...  | 6.77
    ...  | ...    | ...    | ...  | ...  | ...
    pt N | 2.56   | 1.11   | 1    | ...  | 1.23

    Note that unlike in the data DataFrame, the column headers do matter
    as they are used to generate the titles for any plots. Note also that the
    order of rows should match that of the data DataFrame. Finally, labels may
    be continuous, real numbers (as in the height column above) or integers
    (as in the sex column above); however, they MUST BE NUMBERS. Thus, 
    a qualitative label such as handedness, which we would normally label as
    either left or right, must be labelled as 0 or 1. 

    Reducer:
    The reducer is the object that will be used to perform the dimensionality
    reduction on our data. This can be any sklearn pipeline, UMAP object or
    many more! If it has a fit_transform method that takes in data and spits
    out data, then it will work. Useful techniques can be found in the 
    sklearn.manifold module. My recommended reducers can be generated using the
    helper functions in this module. 

    Simple Use:
    Once you have initialised your Visualiser with data, labels and a reducer,
    calling the visualise method will reduce your data and plot it in the 
    correct number of dimensions (1, 2 or 3) for you. It's as easy as that!

    Less Simple Use (But Still Pretty Simple):
    This class also has some other methods that can be called, and the comments
    on each describe what they do. 
    '''

    def __init__(self, data, labels, reducer):

        self._labels = None
        self._data = None
        self._result = None

        self.set_data(data)
        self.set_labels(labels)
        self.set_reducer(reducer)

    def set_data(self, data):

        # Check if data are dataframe
        if not isinstance(data, pd.DataFrame):
            print("Data have not been provided as a pandas DataFrame",
                  "and will be cast to one. Errors may occur.")
            data = pd.DataFrame(data = data)

        self._data = data

        if not self.data_consistent():
            print("The data and labels DataFrames are different sizes.")

    def get_data(self):
        return self._data

    def set_labels(self, labels):
        # Check if labels are dataframe
        if not isinstance(labels, pd.DataFrame):
            # Convert to dataframe with default titles
            print("Labels have not been provided as a pandas DataFrame",
                  "and will be cast to one. Default label names will be",
                  "generated and errors may occur.")
            labels = pd.DataFrame(data = labels)

        self._labels = labels
        
        if not self.data_consistent():
            print("The data and labels DataFrames are different sizes.")

    def get_labels(self):
        return self._labels

    def set_reducer(self, reducer):
        # Check that the reducer has the required methods
        fit = getattr(reducer, "fit", None)
        if not callable(fit):
            print("Reducer does not possess a fit method")
        fit_transform = getattr(reducer, "fit_transform", None)
        if not callable(fit_transform):
            print("Reducer does not possess a fit_transform method")
        transform = getattr(reducer, "transform", None)
        if not callable(transform):
            print("Reducer does not possess a transform method")

        self._reducer = reducer

    def get_reducer(self):
        return self._reducer

    def set_result(self, result):
        self._result = result

    def get_result(self):
        return self._result

    def data_consistent(self):
        if self._labels is None or self._data is None:
            return True

        # Check that labels and data have same number of points
        numLabels, _ = self._labels.shape
        numDataPts, _ = self._data.shape

        return numLabels == numDataPts

    def fit(self):
        self._reducer.fit(self._data)

    def transform(self):
        self._result = self._reducer.transform(self._data)

    def fit_transform(self):
        self._result = self._reducer.fit_transform(self._data)

    def __plot_result_1d(self, fig, rows, cols, index):
        points = np.size(self._result)

        ax = fig.add_subplot(rows, cols, index + 1)
        ax.scatter(self._result, np.zeros((points, )), c= self._labels[index, :], cmap = 'Spectral', s = 5)

    def __plot_result_2d(self, fig, rows, cols, index):
        # Get labels for plot
        labels = self._labels.iloc[:, index].astype(np.float32)
        max_label = np.amax(labels)
        min_label = np.amin(labels)
        delta_labels = max_label - min_label + 1
        colourBarOn = True if delta_labels <= 30 else False

        ax = fig.add_subplot(rows, cols, index + 1)
        im = ax.scatter(self._result[:, 0], self._result[:, 1], c= self._labels.iloc[:, index].astype(np.float32), cmap = 'Spectral', s = 5)
        if colourBarOn:
            fig.colorbar(im, boundaries = np.arange(min_label, max_label + 2) - 0.5, ax = ax).set_ticks(np.arange(min_label, max_label + 2))
        ax.set_title(self._labels.columns[index])
        plt.gca().set_aspect('equal', 'datalim')

    def __plot_result_3d(self, fig, rows, cols, index):
        labels = self._labels.iloc[:, index]

        # Handle non-numeric and numeric labels
        #if issubclass(labels.dtype.type, int):
        is_numeric = True

        try:
            labels = labels.astype(float)
        except:
            is_numeric = False

        if is_numeric:
            print("Numeric")
            max_label = np.amax(labels)
            min_label = np.amin(labels)
            delta_labels = max_label - min_label + 1
            c = self._labels.iloc[:, index].astype(np.float32)
            boundaries = np.arange(delta_labels + 1) - 0.5
            ticks = np.arange(delta_labels)
        else:
            print("Non-numeric")
            unique = labels.unique()
            c = np.array([int((unique == label)[0]) for label in labels])
            boundaries = np.arange(unique.size + 1) - 0.5
            ticks = np.arange(unique.size + 1)

        colourBarOn = True
        #colourBarOn = True if delta_labels <= 30 else False

        ax = fig.add_subplot(rows, cols, index + 1, projection='3d')
        im = ax.scatter(self._result[:, 0], self._result[:, 1], self._result[:, 2], c = c, cmap = 'Spectral', s = 5)
        
        if colourBarOn:
            cbar = fig.colorbar(im, boundaries = boundaries, ax = ax)
            cbar.set_ticks(ticks)
            if not is_numeric:
                cbar.set_ticklabels(unique)
        ax.set_title(self._labels.columns[index])

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
        plt.show()

        return fig

    def visualise(self):
        self.fit_transform()
        self.plot_result()






        
        
       
