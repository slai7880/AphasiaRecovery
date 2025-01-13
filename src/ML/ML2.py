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
import warnings
from sklearn.exceptions import ConvergenceWarning


from common import *

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
    

# no fine-tuning
def runFeatureTypeCombinationLOOExperiments(data0, Y, featureTypes, featureTypeCombinations, model, modelArgs = None, mode = "Classification", processes = 1):
    data = {key : data0[key].copy() for key in data0}
    
    M = {"Feature Combination" : featureTypeCombinations}
    for i in range(len(Y)):
        M[i] = [0] * len(featureTypeCombinations)
    for i in range(len(featureTypeCombinations)):
        print("Feature type combination: " + str(featureTypeCombinations[i]))
        startTime = time.time()
        X = [data[f] for f in featureTypeCombinations[i]]
        X = np.hstack(X)
        args = []
        for test in range(len(X)):
            train = [j for j in range(len(X)) if j != test]
            XTrain, XTest, YTrain, YTest = X[train], X[[test]], Y[train], Y[test]
            if model == "SVM":
                clf = SVC(gamma = "auto")
            elif model == "RF":
                clf = RFC(n_estimators = 400, max_depth = 20, min_samples_split = 2, min_samples_leaf = 2, random_state = 4)
            elif model == "AdaBoost":
                clf = ABC(base_estimator = DTC(max_depth = 1), n_estimators = 100, learning_rate = 1, random_state = 4)
            elif model == "GradientBoosting":
                clf = GBC(loss = "deviance", learning_rate = 0.1, n_estimators = 100, subsample = 1.0, max_depth = 3, random_state = 4)
            if modelArgs:
                clf.set_params(**modelArgs)
            args.append((XTrain, YTrain, XTest, clf))
        YPredicts = None
        if processes > 1:
            pool = Pool(processes)
            YPredicts = pool.map(runTrainTest, args)
            pool.close()
            pool.join()
        else:
            YPredicts = [runTrainTest(a) for a in args]
        for test in range(len(X)):
            M[test][i] = YPredicts[test][0]
        endTime = time.time()
        runtime = np.round((endTime - startTime) / 60, 2)
        print("Time elapsed = " + str(runtime) + ".\n")
    return M
    

# no fine-tuning
def runSingleFeatureTypeLOOExperiments(X, Y, features, featuresRanked, model, mode, processes):
    K = len(features)
    M = {"K" : [i for i in range(K)]}
    for i in range(len(Y)):
        M[i] = [0] * K
    XOriginal = X
    for k in trange(1, len(featuresRanked) + 1):
        indices = [i for i in range(len(features)) if features[i] in featuresRanked[:k]]
        X = XOriginal[:, indices]
        print(X.shape)
        args = []
        for test in range(len(X)):
            train = [j for j in range(len(X)) if j != test]
            XTrain, XTest, YTrain, YTest = X[train], X[[test]], Y[train], Y[test]
            if model == "SVM":
                clf = SVC(gamma = "auto")
            else:
                clf = RFC(n_estimators = 400, max_depth = 20, min_samples_split = 2, min_samples_leaf = 2, random_state = 4)
            args.append((XTrain, YTrain, XTest, clf))
        YPredicts = None
        if processes > 1:
            pool = Pool(processes)
            YPredicts = pool.map(runTrainTest, args)
            pool.close()
            pool.join()
        else:
            YPredicts = [runTrainTest(a) for a in args]
        for test in range(len(X)):
            M[test][k - 1] = YPredicts[test][0]
    return M


    

def runExperiments(model, processes, includeRS):
    print("Model = " + model + ", processes = " + str(processes))
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = False)
    
    featureTypes = ["AQ", "CS", "DM", "LS", "PS_W", "FA", "PS_G", "RS"]
    data0 = {key : data[key].copy() for key in featureTypes} # make a copy, just in case
    
    featureTypeCombinations = []
    for i in range(1, len(featureTypes) + 1):
        featureTypeCombinations += list(combinations(featureTypes, i))
    
    Y = data["TX_0.25"]
    counts = {}
    for y in Y:
        counts[y] = 1 + counts.get(y, 0)
    print("Label distribution : " + str(counts))
    
    mode = "Classification"
    
    targetFeatureType = "RS"
    statsMethod = "Correlation"
    rankingColumn = "Global"
    # rankingColumn = "LOO Median"
    features = featuresRaw[FEATURE_TYPE_TO_FEATURES[targetFeatureType]]
    dfRanking = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", "Statistics", targetFeatureType, "RankingsByAbsolute" + re.sub(" ", "", statsMethod) + ".csv"))
    featuresRanked = [eval(dfRanking[rankingColumn][i])[0] for i in range(dfRanking.shape[0])]
    
    
    if includeRS is None:
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations]
    elif includeRS == "0":
        # all other feature types
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations if not targetFeatureType in combo]
        warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)
    else:
        # target and others, K RS features fixed
        K = int(includeRS)
        featuresTopK = featuresRanked[:K]
        indices = [i for i in range(len(features)) if features[i] in featuresTopK]
        data0[targetFeatureType] = data0[targetFeatureType][:, indices]
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations if targetFeatureType in combo]
    
    
    # feature type combination, no fine tuning
    
    outputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_v7", "Transformed", mode, "Untuned", "LOO", "TopKExperiments", targetFeatureType, statsMethod, rankingColumn, model)
    pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
    M = runFeatureTypeCombinationLOOExperiments(data0, Y, features, featureTypeCombinations2, model)
    dfOut = pandas.DataFrame(M)
    dfOut.to_csv(os.path.join(outputDir, model + "_Top" + str(K) + "RS_Predictions.csv"), index = False)


if __name__ == "__main__":
    model = "RF"
    processes = 1
    includeRS = "21"
    if "-m" in sys.argv:
        model = sys.argv[sys.argv.index("-m") + 1]
    if "-p" in sys.argv:
        processes = int(sys.argv[sys.argv.index("-p") + 1])
    if "-rs" in sys.argv:
        includeRS = sys.argv[sys.argv.index("-rs") + 1]
    # runExperiments(model, processes, includeRS)