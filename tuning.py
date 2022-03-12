import os
import time
from tabnanny import verbose
import pandas as pd
import numpy as np
from tune_sklearn import TuneSearchCV
# import dask_searchcv as dcv
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.model_selection import ParameterGrid

from dask.distributed import Client
import joblib

client = Client(processes=False)  

df = pd.read_csv(r'E:\Others\Wyzant\Elly\Streamlit Website\data\feature_set.csv')


tscv = TimeSeriesSplit(gap=0, n_splits=113)

parameters = {'alpha':list(np.linspace(0.0,1.0,2)), 'binarize':list(np.linspace(0.0,1.0,2)), 'fit_prior':[True, False]}
print(len(ParameterGrid(parameters)))

with joblib.parallel_backend('dask'):

       
    clf = GridSearchCV(estimator=BernoulliNB(), param_grid=parameters, cv=tscv, verbose=1, n_jobs=-1)

    start = time.time()

    clf.fit(df.drop('Class',axis=1),df['Class'])

    end = time.time()

print("Tune Fit Time:", end - start)

print('Best parameters for Bernoulli Naive Bayes :\n', clf.best_params_)


