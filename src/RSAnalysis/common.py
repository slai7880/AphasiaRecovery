import sys, os, pathlib
import numpy as np
import scipy as sp
import pandas
import matplotlib.pyplot as plt

REVERSE_TRANSFORM = False
MATRIX_TYPES = ["Bivariate", "Semipartial"]
TARGETS = ["AQ", "TX"]
LABEL_COLUMN_INDEX = 1

# paths
CROSS_SITE_DATASET = os.path.join("..", "..", "Hariri", "Cross-site_data")
DATA_FILENAME = "compiled_dataset_RSbivariate_without_controls_v6.xlsx"
PATH_TO_FIGURES = os.path.join("..", "..", "Figures")
PATH_TO_RESULTS = os.path.join("..", "..", "results")
PATH_TO_TABLES = os.path.join("..", "..", "Tables")


FIGURE_DIR = os.path.join(PATH_TO_FIGURES, "RSExperiments")
TABLE_DIR = os.path.join(PATH_TO_TABLES, "RSExperiments")
if REVERSE_TRANSFORM:
    FIGURE_DIR = os.path.join(FIGURE_DIR, "ReverseTransformed")
    TABLE_DIR = os.path.join(TABLE_DIR, "ReverseTransformed")
else:
    FIGURE_DIR = os.path.join(FIGURE_DIR, "Transformed")
    TABLE_DIR = os.path.join(TABLE_DIR, "Transformed")

# plotting
COLOR_MAP = plt.get_cmap("cool")

def getCorrelationData(filepath, labelColumnIndex = LABEL_COLUMN_INDEX, reverseTransform = REVERSE_TRANSFORM):
    df = pandas.read_excel(filepath)
    IDs = df["participant"].values.tolist()
    Y = df.iloc[:, labelColumnIndex].values
    XHeaders = df.columns[labelColumnIndex + 1 :]
    X = df.iloc[:, labelColumnIndex + 1 :].values
    if reverseTransform:
        X = np.tanh(X)
    return IDs, X, XHeaders, Y