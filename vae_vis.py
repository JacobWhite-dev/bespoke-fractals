import numpy as np
import pandas as pd
import visualiser
import metrics
import umap
import ast

'''
Data headings are:
time_point,
filename,
ID,
laterality,
cohort,
version,
KL_grade,
mean,
std
'''

# Load data
results = pd.read_csv('result256.csv')
print("Data Loaded")

# Reducer
dim = 512
reducer = umap.UMAP(n_components = 2, metric = metrics.fast_bhattacharyya_mvn, 
                    metric_kwds = {'dim': dim}, verbose = True, )

# Labels
labels = results[['time_point', 'ID', 'laterality', 'cohort', 'version', 'KL_grade']]

cohorts = {'C': 1, 'E': 2}
lateralities = {'Left': 0, 'Right': 1}

def cohort_rename(cohort):
    return cohorts.get(cohort, 0)

def laterality_rename(laterality):
    return lateralities.get(laterality, -1)

cohort_rename_vec = np.vectorize(cohort_rename)
laterality_rename_vec = np.vectorize(laterality_rename)

labels = labels.apply(lambda x: cohort_rename_vec(x) if x.name == 'cohort' else x)
labels = labels.apply(lambda x: laterality_rename_vec(x) if x.name == 'laterality' else x)

print(labels)

# Data
means = np.array([ast.literal_eval(row[0])[0] for row in results[['mean']].values])
stds = np.array([ast.literal_eval(row[0])[0] for row in results[['std']].values])
vars = np.power(stds, 2)
data = np.concatenate((means, vars), axis = 1)

print("Preprocessing Done")

# Visualiser
vis = visualiser.Visualiser(data, labels, reducer)
vis.fit_transform()
vis.plot_result()