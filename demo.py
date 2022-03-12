import numpy as np
from dask.distributed import Client
import time
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, ParameterGrid
from sklearn.svm import SVC

client = Client(processes=False)             # create local cluster

digits = load_digits()

param_space = {'alpha':np.linspace(0.0,1.0,1), 'binarize':np.linspace(0.0,1.0,1), 'fit_prior':[True, False]}
model = SVC(kernel='rbf')
tscv = TimeSeriesSplit(gap=0, n_splits=113)

search = GridSearchCV(model, param_space, cv=tscv, verbose=10)

# with joblib.parallel_backend('dask'):
#     start = time.time()
#     search.fit(digits.data, digits.target)
#     end = time.time()

# print("Tune Fit Time:", end - start)

print(len(ParameterGrid(param_space)))
print(param_space['alpha'])