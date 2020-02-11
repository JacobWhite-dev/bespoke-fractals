import pandas as pd
import visualiser as vis
from collections import defaultdict
dates_dict = defaultdict(list)
import umap
import numpy as np
from vaeumap import VAEUMAP
from metrics import fast_bhattacharyya_mvn
from tabulate import tabulate

# Loading data
data = pd.read_csv('code.csv', header = None, usecols = [i for i in range(1, 513)])
data[256:] = data[256:].applymap(lambda x: x * x)
filenames_df = pd.read_csv('code.csv', header = None, usecols = [0])
filenames = filenames_df.values[:, 0]

labels = defaultdict(list)

for file in filenames:
    pairs = file.split("##")[:-1]
    for pair in pairs:
        key, value = pair.split("_")
        labels[key].append(value)

# Represent cohorts numerically
#lat_nums = {"L": 0, "R": 1}
#labels["lat"] = map(lambda x: lat_nums.get(x, 2), labels["lat"])
#cohort_nums = {"C": 0, "E": 1}
#labels["cohort"] = map(lambda x: cohort_nums.get(x, 2), labels["cohort"])
labels = pd.DataFrame(data = labels)
print(labels)

# Visualise
dim = 256

reducer = VAEUMAP(dim, random_state = 2, n_components = 2, verbose = True)
visualiser = vis.Visualiser(data, labels, reducer)
#visualiser.fit_transform()
visualiser.visualise()

# Identifying small blob (x is greater than 10 on all)
#result = visualiser.get_result()
#pts = np.argwhere(result[:, 0] > 10)
#print(pts)
#print(filenames[pts])




