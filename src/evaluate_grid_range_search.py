from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing
from skrebate import ReliefF, SURF

# plt.switch_backend('agg')

from common import *

def makeMatrixPlot(M, title,\
                xTickLabels = None, yTickLabels = None,\
                xLabel = "", yLabel = "",\
                threshold = "auto", cmap = plt.cm.Blues, vmin = None, vmax = None,\
                figsize = (8, 6), dpi = 100,\
                integer = True):
    """
    Creates a 2D plot of a numpy matrix.
    Parameters
    ----------
    M : numpy matrix
    title : string
        The title of the plot.
    xTickLabels, yTickLabels : string
        The tick labels on the axis'.
    xLabel, yLabel : string
        The labels of the axis'.
    threshold : int or float
        The threshold where the text color should flip so that the text
        can be seen.
    cmap : matplotlib.pyplot color map object
    integer : boolean
        Indicates whether or not the values in M should be displayed as integers.
    Returns
    -------
    fig, ax
        The objects of matplotlib.pyplot figure.
    """
    if xTickLabels is None:
        xTickLabels = [i for i in range(M.shape[1])]
    if yTickLabels is None:
        yTickLabels = [i for i in range(M.shape[0])]
    fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
    im = ax.imshow(M, interpolation = "nearest", cmap = cmap, vmin = vmin, vmax = vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size = "5%", pad = 0.05)
    ax.figure.colorbar(im, cax = cax)
    ax.set_title(title, fontsize = 16)
    ax.set_xlabel(xLabel, fontsize = 14)
    ax.set_ylabel(yLabel, fontsize = 14)
    ax.set(xticks = np.arange(M.shape[1]), yticks = np.arange(M.shape[0]), xticklabels = xTickLabels, yticklabels = yTickLabels)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
    if threshold == "auto":
        threshold = M.max() / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            color = "white"
            if M[i, j] <= threshold:
                color = "black"
            text = str(M[i, j])
            if integer:
                text = str(int(M[i, j]))
            ax.text(j, i, text, ha = "center", va = "center", color = color)
    fig.tight_layout()
    return fig, ax

def evaluateSVMParamSearchExperiments():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["TX_0.25"]
    
    grid = {"C" : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
            "gamma" : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]}
    params = [*grid]
    F1ScoresOptimal = {"RBF" : [], "Linear" : []}
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "SVMParameterSearch")
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    include_RS = 20
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO", "SVMParameterSearch", "RS" + str(include_RS))

    # grid search
    df = pandas.read_csv(os.path.join(inputDir, "RBFSVM_Grid_Results.csv"))
    YPredicts = np.zeros((df.shape[0], len(YTrue)), dtype = "int")
    F1Scores = np.zeros((len(grid["C"]), len(grid["gamma"])), dtype = "float")
    for i in range(YPredicts.shape[0]):
        for j in range(YPredicts.shape[1]):
            accuracy = df["split" + str(j) + "_test_score"][i]
            if accuracy == 0:
                if YTrue[j] == 0:
                    YPredicts[i, j] = 1
                else:
                    YPredicts[i, j] = 0
            else:
                YPredicts[i, j] = YTrue[j]
        F1Scores[grid["C"].index(df["param_C"][i]), grid["gamma"].index(df["param_gamma"][i])] = f1_score(YTrue, YPredicts[i, :])
    F1Scores = np.round(F1Scores, 2)
    
    fig, ax = makeMatrixPlot(M = F1Scores, title = "RBF SVM F1 Scores over Parameter Grid",\
                xTickLabels = grid["gamma"], yTickLabels = grid["C"],\
                xLabel = "gamma", yLabel = "C",\
                threshold = "auto", cmap = plt.cm.Blues, vmin = 0, vmax = 1,\
                figsize = (16, 16), dpi = 100,\
                integer = False)
    fig.savefig(os.path.join(figDir, "F1OverRBFSVMParamGridRS" + str(include_RS) + ".png"))
    plt.close(fig)
    
    
    # linear SVM

    df = pandas.read_csv(os.path.join(inputDir, "LinearSVM_Grid_Results.csv"))
    YPredicts = np.zeros((df.shape[0], len(YTrue)), dtype = "int")
    F1Scores = np.zeros(len(grid["C"]), dtype = "float")
    for i in range(YPredicts.shape[0]):
        for j in range(YPredicts.shape[1]):
            accuracy = df["split" + str(j) + "_test_score"][i]
            if accuracy == 0:
                if YTrue[j] == 0:
                    YPredicts[i, j] = 1
                else:
                    YPredicts[i, j] = 0
            else:
                YPredicts[i, j] = YTrue[j]
        F1Scores[grid["C"].index(df["param_C"][i])] = f1_score(YTrue, YPredicts[i, :])
    F1Scores = np.round(F1Scores, 2)
    
    fig, ax = plt.subplots(figsize = (12, 10))
    ax.plot(F1Scores)
    ax.set_title("Linear SVM F1 Scores over C", fontsize = 16)
    ax.set_xlabel("C")
    fig.savefig(os.path.join(figDir, "F1OverLinearSVMParamGridRS" + str(include_RS) + ".png"))
    plt.close(fig)

def evaluateRFParamSearchExperiments():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["TX_0.25"]
    
    grid = {"criterion" : ["gini", "entropy"],
            "max_depth" : [4, 8, 12, 16, 20, 24, 28, 32, 36],
            "max_features" : ["log2", "sqrt", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]}
    criteria = grid["criterion"]
    params = [*grid]
    include_RS_choices = [str(i) for i in range(1, 101)]
    F1ScoresOptimal = {"gini" : [], "entropy" : []}
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "RFParameterSearch")
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    include_RS = 20
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO", "RFParameterSearch", "RS" + str(include_RS))
    
    # grid search
    df = pandas.read_csv(os.path.join(inputDir, "RF_Grid_Results.csv"))
    YPredicts = np.zeros((df.shape[0], len(YTrue)), dtype = "int")
    F1Scores = {"gini" : np.zeros((len(grid["max_depth"]), len(grid["max_features"])), dtype = "float"),
                "entropy" : np.zeros((len(grid["max_depth"]), len(grid["max_features"])), dtype = "float")}
    for i in range(YPredicts.shape[0]):
        for j in range(YPredicts.shape[1]):
            accuracy = df["split" + str(j) + "_test_score"][i]
            if accuracy == 0:
                if YTrue[j] == 0:
                    YPredicts[i, j] = 1
                else:
                    YPredicts[i, j] = 0
            else:
                YPredicts[i, j] = YTrue[j]
        if df["param_criterion"][i] == "gini":
            F1Scores["gini"][grid["max_depth"].index(df["param_max_depth"][i]), grid["max_features"].index(df["param_max_features"][i])] = f1_score(YTrue, YPredicts[i, :])
        elif df["param_criterion"][i] == "entropy":
            F1Scores["entropy"][grid["max_depth"].index(df["param_max_depth"][i]), grid["max_features"].index(df["param_max_features"][i])] = f1_score(YTrue, YPredicts[i, :])
    for i in range(len(criteria)):
        F1Scores[criteria[i]] = np.round(F1Scores[criteria[i]], 2)
    
    fig, axes = plt.subplots(1, 2, figsize = (32, 16))
    for k in range(len(criteria)):
        M = F1Scores[criteria[k]]
        im = axes[k].imshow(M, interpolation = "nearest", cmap = plt.cm.Blues, vmin = 0.0, vmax = 1.0)
        axes[k].set_title("RF (" + criteria[k] + ") F1 Scores over Pamameter Grid", fontsize = 16)
        axes[k].set_xlabel("Max Features", fontsize = 14)
        axes[k].set_ylabel("Max Depth", fontsize = 14)
        axes[k].set(xticks = np.arange(M.shape[1]), yticks = np.arange(M.shape[0]), xticklabels = grid["max_features"], yticklabels = grid["max_depth"])
        plt.setp(axes[k].get_xticklabels(), rotation = 45, ha = "right", rotation_mode = "anchor")
        threshold = 0.5
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                color = "white"
                if M[i, j] <= threshold:
                    color = "black"
                text = str(M[i, j])
                axes[k].text(j, i, text, ha = "center", va = "center", color = color)
    fig.subplots_adjust(right = 0.8)
    cbar_ax = fig.add_axes([0.82, 0.246, 0.01, 0.5])
    fig.colorbar(im, cax = cbar_ax)
    fig.savefig(os.path.join(figDir, "F1OverRFParamGridRS" + str(include_RS) + ".png"))
    plt.close(fig)
    
def evaluateAdaBoostParamSearchExperiments():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["TX_0.25"]
    
    grid = {"n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "learning_rate" : [1e-3, 1e-2, 1e-1, 1]}
    params = [*grid]
    include_RS_choices = [str(i) for i in range(1, 101)]
    F1ScoresOptimal = []
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "AdaBoostParameterSearch")
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    include_RS = 20
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO", "AdaBoostParameterSearch", "RS" + str(include_RS))
    
    # grid search
    df = pandas.read_csv(os.path.join(inputDir, "AdaBoost_Grid_Results.csv"))
    YPredicts = np.zeros((df.shape[0], len(YTrue)), dtype = "int")
    F1Scores = np.zeros((len(grid["n_estimators"]), len(grid["learning_rate"])), dtype = "float")
    for i in range(YPredicts.shape[0]):
        for j in range(YPredicts.shape[1]):
            accuracy = df["split" + str(j) + "_test_score"][i]
            if accuracy == 0:
                if YTrue[j] == 0:
                    YPredicts[i, j] = 1
                else:
                    YPredicts[i, j] = 0
            else:
                YPredicts[i, j] = YTrue[j]
        F1Scores[grid["n_estimators"].index(df["param_n_estimators"][i]), grid["learning_rate"].index(df["param_learning_rate"][i])] = f1_score(YTrue, YPredicts[i, :])
    F1Scores = np.round(F1Scores, 2)
    F1ScoresOptimal.append(F1Scores.max())
    
    fig, ax = makeMatrixPlot(M = F1Scores, title = "AdaBoost F1 Scores over Parameter Grid",\
                xTickLabels = grid["learning_rate"], yTickLabels = grid["n_estimators"],\
                xLabel = "Learning Rate", yLabel = "Number of Trees",\
                threshold = "auto", cmap = plt.cm.Blues, vmin = 0, vmax = 1,\
                figsize = (10, 14), dpi = 100,\
                integer = False)
    fig.savefig(os.path.join(figDir, "F1OverAdaBoostParamGridRS" + str(include_RS) + ".png"))
    plt.close(fig)
    

def evaluateSVMRFAdaBoost():
    include_RS_choices = [str(i) for i in range(1, 101)]
    F1ScoresOptimal_SVM = evaluateSVMParamSearchExperiments()
    F1ScoresOptimal_RF = evaluateRFParamSearchExperiments()
    F1ScoresOptimal_AdaBoost = evaluateAdaBoostParamSearchExperiments()
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned")
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    fig, axes = plt.subplots(3, 1, figsize = (24, 18))
    
    for key in F1ScoresOptimal_SVM:
        axes[0].plot(F1ScoresOptimal_SVM[key], label = key)
    axes[0].set_title("Optimal SVM F1 Scores with Top K RS Features", fontsize = 16)
    
    for key in F1ScoresOptimal_RF:
        axes[1].plot(F1ScoresOptimal_RF[key], label = key)
    axes[1].set_title("Optimal RF F1 Scores with Top K RS Features", fontsize = 16)
    
    axes[2].plot(F1ScoresOptimal_AdaBoost)
    axes[2].set_title("Optimal AdaBoost F1 Scores with Top K RS Features", fontsize = 16)
    
    for i in range(len(axes)):
        axes[i].set_xlabel("K", fontsize = 14)
        axes[i].set_xticks([i for i in range(len(include_RS_choices))])
        axes[i].set_xticklabels(include_RS_choices)
        axes[i].grid(True)
        axes[i].legend(prop = {"size" : 14})
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "ParameterRangeSearchOptimalF1Scores.png"))
    plt.close(fig)

if __name__ == "__main__":
    evaluateSVMParamSearchExperiments()
    evaluateRFParamSearchExperiments()
    # evaluateAdaBoostParamSearchExperiments()
    # evaluateSVMRFAdaBoost()