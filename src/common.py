import sys, os, pathlib, re, time
import numpy as np
import scipy as sp
import pandas
from pprint import pprint
from multiprocessing import Pool
import multiprocessing
import pickle
from itertools import combinations, product

import sklearn.feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import trange

PATH_TO_PROJECT = os.path.join(os.sep, "scratch", "lais823", "BU", "Rehab")
CROSS_SITE_DATASET = os.path.join(PATH_TO_PROJECT, "Hariri", "Cross-site_data")
DATA_VERSION = "v7"
DATA_FILENAME = "compiled_dataset_RSbivariate_without_controls_" + DATA_VERSION + ".xlsx"
PATH_TO_FIGURES = os.path.join(PATH_TO_PROJECT, "Figures")
PATH_TO_RESULTS = os.path.join(PATH_TO_PROJECT, "results")
PATH_TO_TABLES = os.path.join(PATH_TO_PROJECT, "Tables")
PATH_TO_CACHE = os.path.join(PATH_TO_PROJECT, "Cache")

PATH_TO_RS_DATA = os.path.join(CROSS_SITE_DATASET, "resting_state_data")

FEATURE_TYPE_TO_FEATURES = {"PS_G" : "percent_spared_in_gray_matter", "PS_W" : "percent_spared_in_white_matter", "RS" : "restingstate_bivariate_correlations"}

def getData(filepath, header = [0, 1], normalize = False):
    df = pandas.read_excel(filepath, header = header)
    df[('behavioral', "cs1_bd")].fillna(df[('behavioral', "cs1_bd")].median(), inplace = True)
    df[('behavioral', "cs2_bd")].fillna(df[('behavioral', "cs2_bd")].median(), inplace = True)
    data = {}
    data["participant"] = df.iloc[:, 0].values
    data["AQ"] = df[('behavioral', "wab_aq_bd")].values[:, np.newaxis]
    data["CS"] = df[[('behavioral', "cs1_bd"), ('behavioral', "cs2_bd")]].values
    data["DM"] = df["demographic_info"].values
    data["LS"] = df["lesion_size"].values
    data["PS_W"] = df["percent_spared_in_white_matter"].values
    featuresFAL = ["fa_avg_ccmaj", "fa_avg_ccmin", "fa_avg_lifof", "fa_avg_lilf", "fa_avg_lslf", "fa_avg_lunc", "fa_avg_larc"]
    featuresFAR = ["fa_avg_rifof", "fa_avg_rilf", "fa_avg_rslf", "fa_avg_runc", "fa_avg_rarc"]
    FA_L = df["average_FA_values"][featuresFAL].fillna(0).values
    FA_R = df["average_FA_values"][featuresFAR].fillna(df["average_FA_values"][featuresFAR].mean()).values
    data["FA"] = np.hstack((FA_L, FA_R))
    data["PS_G"] = df["percent_spared_in_gray_matter"].values
    data["RS"] = df["restingstate_bivariate_correlations"].values
    
    data["TX"] = df[('behavioral', "difference_post_pre_bd")].values
    data["TX_Z"] = df[('behavioral', "zscorepost_minus_zscorepre_bd")].values
    data["TX_standardized"] = (data["TX_Z"] >= 0).astype("int")
    data["TX_0.25"] = df[('behavioral', "tx_change_categorical_0.25_bd")].values
    data["TX_median"] = df[('behavioral', "tx_change_categorical_median_bd")].values
    
    featuresRaw = {}
    for col in df:
        if not col[0] in featuresRaw:
            featuresRaw[col[0]] = []
        featuresRaw[col[0]].append(col[1])
    features = {"AQ" : ["wab_aq_bd"], "CS" : ["cs1_bd", "cs2_bd"],
                "DM" : featuresRaw["demographic_info"], "LS" : featuresRaw["lesion_size"], "FA" : featuresRaw["average_FA_values"],
                "PS_G" : featuresRaw["percent_spared_in_gray_matter"],
                "PS_W" : featuresRaw["percent_spared_in_white_matter"],
                "RS" : featuresRaw["restingstate_bivariate_correlations"]}
    
    if normalize:
        data["AQ"] = (data["AQ"] - data["AQ"].min(axis = 0)) / (data["AQ"].max(axis = 0) - data["AQ"].min(axis = 0))
        data["DM"] = (data["DM"] - data["DM"].min(axis = 0)) / (data["DM"].max(axis = 0) - data["DM"].min(axis = 0))
        data["LS"] = (data["LS"] - data["LS"].min(axis = 0)) / (data["LS"].max(axis = 0) - data["LS"].min(axis = 0))
        data["RS"] = (data["RS"] - data["RS"].min(axis = 0)) / (data["RS"].max(axis = 0) - data["RS"].min(axis = 0))
    
    return data, features
    


def getFeatureCombinations(features, featureNames):
    XAll = {}
    for i in range(1, len(features) + 1):
        namesSet = list(combinations(featureNames, i))
        for names in namesSet:
            XAll[" + ".join(names)] = np.nan_to_num(np.hstack([features[name] for name in names]))
    return XAll