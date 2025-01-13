import sys, os, pathlib, re, time
import numpy as np
import scipy as sp
import pandas
from pprint import pprint
from multiprocessing import Pool
import multiprocessing
import pickle
from itertools import combinations, product
from tqdm import trange

import sklearn.feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.pipeline import Pipeline


class CustomizedGridSearchCV:
    def __init__(self, estimator, param_grid, scoring, cv, refit = True, n_jobs = 1):
        self.estimator = estimator
        self.grid = param_grid
        self.gridDimensions = [len(self.grid[p]) for p in self.grid]
        self.parameters = [*self.grid]
        self.paramIndexCombos = list(product(*[[i for i in range(d)] for d in self.gridDimensions]))
        self.scoring = scoring
        self.splits = list(cv)
        self.refit = refit
        self.processes = n_jobs
        self.optimalClf = None
    
    def runTrainTest(self, args):
        XTrain, XTest, YTrain, parameterValues = args
        clf = clone(self.estimator)
        clf.set_params(**parameterValues)
        clf.fit(XTrain, YTrain)
        return clf.predict(XTest)
    
    def fit(self, X, Y):
        self.CVScoreGrid = np.zeros(self.gridDimensions)
        self.CVPredictionGrid = np.zeros(self.gridDimensions + [len(Y)])
        for indices in self.paramIndexCombos:
            parameterValues = {self.parameters[i] : self.grid[self.parameters[i]][indices[i]] for i in range(len(indices))}
            YTrue, YPredict = np.array([]), np.array([])
            args = []
            for train, test in self.splits:
                XTrain, XTest, YTrain, YTest = X[train], X[test], Y[train], Y[test]
                YTrue = np.concatenate((YTrue, YTest))
                args.append((XTrain, XTest, YTrain, parameterValues))
            YPredict = None
            if self.processes > 1:
                pool = Pool(self.processes)
                YPredict = np.concatenate(list(pool.map(self.runTrainTest, args)))
                pool.close()
                pool.join()
            else:
                YPredict = np.concatenate([self.runTrainTest(a) for a in args])
            self.CVScoreGrid[indices] = self.scoring(YTrue, YPredict)
            self.CVPredictionGrid[indices] = YPredict
        if self.refit:
            optimalIndices = np.unravel_index(self.CVScoreGrid.argmax(), self.gridDimensions)
            optimalParamValues = {self.parameters[i] : self.grid[self.parameters[i]][optimalIndices[i]] for i in range(len(optimalIndices))}
            self.optimalClf = clone(self.estimator)
            self.optimalClf.set_params(**optimalParamValues)
            self.optimalClf.fit(X, Y)
            
    def predict(self, X):
        if self.optimalClf is None:
            print("The module is not fitted yet. Set refit to be True before calling the fit function")
        else:
            return self.optimalClf.predict(X)


def runTrainTest(args):
    XTrain, YTrain, XTest, clf = args
    clf.fit(XTrain, YTrain)
    return clf.predict(XTest)