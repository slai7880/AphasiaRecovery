from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing
from itertools import combinations
import sklearn.feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, SpectralEmbedding
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import bisect
import random



from common import *


def splitRSData(filepath, outputDir, labelColumn, K = 5, seeds = [4]):
    df = pandas.read_excel(filepath)
    for seed in seeds:
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
        count = 0
        for train, test in skf.split([i for i in range(df.shape[0])], df[labelColumn].values):
            dfTrain, dfTest = df.iloc[train, :], df.iloc[test, :]
            foldDir = os.path.join(outputDir, str(seed), str(count))
            pathlib.Path(foldDir).mkdir(parents = True, exist_ok = True)
            dfTrain.to_csv(os.path.join(foldDir, "train.csv"), index = False)
            dfTest.to_csv(os.path.join(foldDir, "test.csv"), index = False)
            count += 1

def makeTargetCorrelationTuples(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX):
    tuples = {t : {m : [] for m in matrixTypes} for t in targets}
    for i in range(len(targets)):
        for j in range(len(matrixTypes)):
            matrixType, target = matrixTypes[j], targets[i]
            filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
            filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
            _, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
            for k in range(X.shape[1]):
                r, p = sp.stats.pearsonr(np.array(X[:, k].reshape(1, -1).squeeze()), Y)
                tuples[target][matrixType].append((XHeaders[k], r, p, abs(r)))
    return tuples
    
def makeTargetCorrelationHistograms(targetCorrelationTuples, matrixTypes = MATRIX_TYPES, targets = TARGETS, threshold = 0):
    histograms = {t : {m : None for m in matrixTypes} for t in targets}
    for i in range(len(targets)):
        for j in range(len(matrixTypes)):
            matrixType, target = matrixTypes[j], targets[i]
            tuples = targetCorrelationTuples[target][matrixType]
            tuplesSorted = sorted(tuples, reverse = True, key = lambda x : x[-1])
            features = [tuplesSorted[k][0] for k in range(len(tuplesSorted))]
            values = [tuplesSorted[k][-1] for k in range(len(tuplesSorted))]
            colors = []
            for k in range(len(tuplesSorted)):
                if tuplesSorted[k][1] >= 0:
                    colors.append("m")
                else:
                    colors.append("c")
            histograms[target][matrixType] = {"features" : features, "values" : values, "colors" : colors}
    for j in range(len(matrixTypes)):
        matrixType = matrixTypes[j]
        featuresToDrop = []
        for k in range(len(targetCorrelationTuples[targets[0]][matrixType])):
            tupleAQ = targetCorrelationTuples[targets[0]][matrixType][k]
            tupleTX = targetCorrelationTuples[targets[1]][matrixType][k]
            if tupleAQ[3] < threshold or tupleTX[3] < threshold:
                featuresToDrop.append(tupleAQ[0])
        for target in targets:
            for k in range(len(histograms[target][matrixType]["values"])):
                if histograms[target][matrixType]["features"][k] in featuresToDrop:
                    histograms[target][matrixType]["values"][k] = 0
    return histograms

def getDTMaxDepth(tree):
    nodeDepth = np.zeros(shape = tree.node_count, dtype = np.int64)
    stack = [(0, -1)]
    while len(stack) > 0:
        id, parentDepth = stack.pop()
        nodeDepth[id] = parentDepth + 1
        if tree.children_left[id] != tree.children_right[id]:
            stack.append((tree.children_left[id], parentDepth + 1))
            stack.append((tree.children_right[id], parentDepth + 1))
    return nodeDepth.max()

def runTrainTest(args):
    XTrain, YTrain, XTest, clf = args
    clf.fit(XTrain, YTrain)
    return clf.predict(XTest)
    
def getClassifier(clfName):
    if clfName == "SVM(RBF)":
        return SVC(gamma = "auto")
    elif clfName == "SVM(Linear)":
        return SVC(gamma = "auto", kernel = "linear")
    elif clfName == "RFC(100)":
        return RFC(100, random_state = 4)
    elif clfName == "RFC(200)":
        return RFC(200, random_state = 4)
        
def runClassicalExperiments(XTrain, YTrain, XTest, clfName, resultDir):
    args = (XTrain, YTrain, XTest, getClassifier(clfName))
    YPredict = runTrainTest(args)
    np.save(os.path.join(resultDir, "YPredict.npy"), YPredict)
    
def runRFEExperiments(XTrain, YTrain, XTest, clfName, resultDir):
    
    clfBase = getClassifier(clfName)
    rfe = fs.RFECV(estimator = clfBase, cv = StratifiedKFold(n_splits = 4, shuffle = True), n_jobs = N_PROCESSES)
    rfe.fit(XTrain, YTrain)
    YPredict = rfe.predict(XTest)
    featureRanking = rfe.ranking_
    np.save(os.path.join(resultDir, "YPredict.npy"), YPredict)
    np.save(os.path.join(resultDir, "FeatureRanking.npy"), featureRanking)
    
def runMLExperiments(labelColumn, splitDir, outputDir, experimentType):
    clfNames = None
    if experimentType == "classical":
        clfNames = ["SVM(RBF)", "SVM(Linear)", "RFC(100)", "RFC(200)"]
    else:
        clfNames = ["SVM(Linear)", "RFC(100)", "RFC(200)"]
    for seed in os.listdir(splitDir):
        seedDir = os.path.join(splitDir, seed)
        for fold in os.listdir(seedDir):
            foldDir = os.path.join(seedDir, fold)
            dfTrain = pandas.read_csv(os.path.join(foldDir, "train.csv"))
            dfTest = pandas.read_csv(os.path.join(foldDir, "test.csv"))
            XTrain, YTrain = dfTrain.iloc[:, 2:].values, dfTrain[labelColumn].values
            XTest, YTest = dfTest.iloc[:, 2:].values, dfTest[labelColumn].values
            for clfName in clfNames:
                resultDir = os.path.join(outputDir, seed, experimentType, clfName, fold)
                pathlib.Path(resultDir).mkdir(parents = True, exist_ok = True)
                if experimentType == "classical":
                    runClassicalExperiments(XTrain, YTrain, XTest, clfName, resultDir)
                else:
                    runRFEExperiments(XTrain, YTrain, XTest, clfName, resultDir)

      
def runRegressionExperiments(threshold, matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX, reverseTransform = REVERSE_TRANSFORM):
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    targetCorrelationTuples = makeTargetCorrelationTuples(matrixTypes, targets, labelColumnIndex)
    
    
    thresholdMode = "Individual"
    
    outputDir = os.path.join(TABLE_DIR, "Regression", "Finetuned")
    pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
    
    NFolds = 5
    NCategories = 5
    skf = StratifiedKFold(n_splits = NFolds, shuffle = True, random_state = 4)
    for i in range(len(targets)):
        for j in range(len(matrixTypes)):
            matrixType, target = matrixTypes[j], targets[i]
            filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
            filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
            IDs, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
            if target == "TX":
                Y *= 100.0
            
            YPredicts = np.array([None for _ in range(X.shape[0])], dtype = float)
            # YPredicts = {f : np.array([[None] * len(thresholds) for _ in range(X.shape[0])], dtype = float) for f in fractions}
            for test in range(X.shape[0]):
                train = [k for k in range(X.shape[0]) if k != test] 
                XTrain, XTest, YTrain, YTest = X[train, :], X[test, :].reshape(1, -1), Y[train], Y[test]
                
                # sort the target values, and put them into bins so that stratified sampling can apply
                tuples = [(k, YTrain[k]) for k in range(len(YTrain))]
                tuples.sort(key = lambda x : x[1])
                labels = [] # use this as target in skf.split
                cap = len(tuples) // NCategories
                counter = 0
                for k in range(len(tuples)):
                    labels.append(counter)
                    if (k + 1) % cap == 0:
                        counter += 1
                
                # reorder the training set, so that the order matches labels
                tuples2 = [(tuples[k][0], labels[k]) for k in range(len(tuples))]
                random.Random(4).shuffle(tuples2)
                indices = [tuples2[k][0] for k in range(len(tuples2))]
                labels = np.array([tuples2[k][1] for k in range(len(tuples2))])
                XTrain, YTrain = XTrain[indices, :], YTrain[indices]
                
                grid = {"n_estimators" : [20, 30, 40, 50, 60, 70, 80],
                        "criterion" : ["mae"],
                        "max_depth" : [2, 4, 6, 8, 10, 12], # make them smaller, as the number of samples is small
                        "min_samples_split" : [0.1, 0.2, 0.3, 0.4],
                        "random_state" : [4]}
                selectedFeatures = []
                if thresholdMode == "Joint":
                    # select features whose correlations with both targets are greater than threshold
                    targetCorrelationHistograms = makeTargetCorrelationHistograms(targetCorrelationTuples, matrixTypes, targets, threshold = threshold)
                    for k in range(len(targetCorrelationHistograms[target][matrixType]["features"])):
                        if targetCorrelationHistograms[target][matrixType]["values"][k] > 0:
                            selectedFeatures.append(targetCorrelationHistograms[target][matrixType]["features"][k])
                elif thresholdMode == "Individual":
                    # select features whose correlations with the current target values are greather than threshold
                    for k in range(len(targetCorrelationTuples[target][matrixType])):
                        tuple = targetCorrelationTuples[target][matrixType][k]
                        if tuple[3] >= threshold:
                            selectedFeatures.append(tuple[0])
                else:
                    print("Error: applyThresholdOn = " + str(applyThresholdOn))
                    sys.exit(1)
                selectedIndices = [k for k in range(len(XHeaders)) if XHeaders[k] in selectedFeatures]
                if len(selectedIndices) > 0:
                    XTrainReduced = XTrain[:, selectedIndices]
                    XTestReduced = XTest[:, selectedIndices]
                    
                    estimator = RFR()
                    splits = skf.split([k for k in range(len(labels))], labels)
                    
                    '''
                    model = GridSearchCV(estimator = estimator, param_grid = grid, cv = splits, n_jobs = 4, iid = False)
                    model.fit(XTrainReduced, YTrain)
                    YPredict = model.predict(XTestReduced)
                    YPredicts[test] = YPredict[0]
                    '''
                    
                    
                    '''
                    for f in fractions:
                        model = RFR(min_samples_split = f, criterion = "mae")
                        model.fit(XTrainReduced, YTrain)
                        YPredict = model.predict(XTestReduced)
                        YPredicts[f][test][t] = YPredict[0]
                    '''
            
            # dfPredicts = pandas.DataFrame({"prediction" : YPredicts, "participant" : IDs})
            # dfPredicts.to_csv(os.path.join(outputDir, "YPredictsRFR" + matrixType + target + thresholdMode + "Thresholded(" + str(threshold) + ").csv"), index = False)
            
            
            '''
            errors = []
            for k in range(len(fractions)):
                f = fractions[k]
                mae = []
                for t in range(len(thresholds)):
                    if np.isnan(YPredicts[f][:, t]).any():
                        mae.append(None)
                    else:
                        mae.append(mean_absolute_error(Y, YPredicts[f][:, t]))
                errors.append(mae)
            dfErrors = pandas.DataFrame(errors, columns = thresholds)
            dfErrors["min_samples_split"] = fractions
            dfErrors = dfErrors[["min_samples_split"] + thresholds]
            dfErrors.to_csv(os.path.join(outputDir, "MAERFR" + matrixType + target + thresholdMode + "ThresholdedMinSplit.csv"), index = False)
            '''
def runConcatenatedRegressionExperiments(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX, reverseTransform = REVERSE_TRANSFORM):
    optimalThresholds = {"AQ" : {"Bivariate" : 0.33, "Semipartial" : 0.33}, "TX" : {"Bivariate" : 0.31, "Semipartial" : 0.35}}
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    targetCorrelationTuples = makeTargetCorrelationTuples(matrixTypes, targets, labelColumnIndex)
    
    
    thresholdMode = "Individual"
    
    outputDir = os.path.join(TABLE_DIR, "Regression", "Finetuned")
    pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
    
    NFolds = 5
    NCategories = 5
    skf = StratifiedKFold(n_splits = NFolds, shuffle = True, random_state = 4)
    for i in range(len(targets)):
        target = targets[i]
        X, Y = [], None
        for j in range(len(matrixTypes)):
            matrixType = matrixTypes[j]
            filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
            filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
            IDs, Xj, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
            for k in range(len(XHeaders)):
                if targetCorrelationTuples[target][matrixType][k][3] >= optimalThresholds[target][matrixType]:
                    X.append(Xj[:, k].reshape(-1, 1))
            if target == "TX":
                Y *= 100.0
        X = np.concatenate(X, axis = 1)
        YPredicts = np.array([None for _ in range(X.shape[0])], dtype = float)
        # YPredicts = {f : np.array([[None] * len(thresholds) for _ in range(X.shape[0])], dtype = float) for f in fractions}
        for test in range(X.shape[0]):
            train = [k for k in range(X.shape[0]) if k != test] 
            XTrain, XTest, YTrain, YTest = X[train, :], X[test, :].reshape(1, -1), Y[train], Y[test]
            
            # sort the target values, and put them into bins so that stratified sampling can apply
            tuples = [(k, YTrain[k]) for k in range(len(YTrain))]
            tuples.sort(key = lambda x : x[1])
            labels = [] # use this as target in skf.split
            cap = len(tuples) // NCategories
            counter = 0
            for k in range(len(tuples)):
                labels.append(counter)
                if (k + 1) % cap == 0:
                    counter += 1
            
            # reorder the training set, so that the order matches labels
            tuples2 = [(tuples[k][0], labels[k]) for k in range(len(tuples))]
            random.Random(4).shuffle(tuples2)
            indices = [tuples2[k][0] for k in range(len(tuples2))]
            labels = np.array([tuples2[k][1] for k in range(len(tuples2))])
            XTrain, YTrain = XTrain[indices, :], YTrain[indices]
            
            grid = {"n_estimators" : [20, 30, 40, 50, 60, 70, 80],
                    "criterion" : ["mae"],
                    "max_depth" : [2, 4, 6, 8, 10, 12], # make them smaller, as the number of samples is small
                    "min_samples_split" : [0.1, 0.2, 0.3, 0.4],
                    "random_state" : [4]}
            estimator = RFR()
            splits = skf.split([k for k in range(len(labels))], labels)
            model = GridSearchCV(estimator = estimator, param_grid = grid, cv = splits, n_jobs = 4, iid = False)
            model.fit(XTrain, YTrain)
            YPredict = model.predict(XTest)
            YPredicts[test] = YPredict[0]
        
        dfPredicts = pandas.DataFrame({"prediction" : YPredicts, "participant" : IDs})
        dfPredicts.to_csv(os.path.join(outputDir, "YPredictsRFRConcatenated" + target + ".csv"), index = False)
            

def postjobAnalysis(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX, reverseTransform = REVERSE_TRANSFORM):
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    targetCorrelationTuples = makeTargetCorrelationTuples(matrixTypes, targets, labelColumnIndex)
    
    # thresholds = [np.round(i * 0.05, 2) for i in range(10)]
    thresholdMode = "Individual"
    
    '''
    thresholds = np.linspace(0.3, 0.4, 11)
    R2, MAE = {m : {t : [] for t in targets} for m in matrixTypes}, {m : {t : [] for t in targets} for m in matrixTypes}
    for i in range(len(targets)):
        for j in range(len(matrixTypes)):
            matrixType, target = matrixTypes[j], targets[i]
            filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
            filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
            IDs, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
            if target == "TX":
                Y *= 100.0
            
            # merge the tables into one
            
            YPredicts = []
            for threshold in thresholds:
                df = pandas.read_csv(os.path.join(TABLE_DIR, "Regression", "Finetuned", "YPredictsRFR" + matrixType + target + thresholdMode + "Thresholded(" + str(threshold) + ").csv"))
                YPredicts.append(df["prediction"].values.reshape(-1, 1))
            YPredicts = np.hstack(YPredicts)
            dfOut = pandas.DataFrame(YPredicts, columns = thresholds)
            dfOut["participant"] = IDs
            dfOut = dfOut[["participant"] + [t for t in thresholds]]
            dfOut.to_csv(os.path.join(TABLE_DIR, "Regression", "Finetuned", "YPredictsRFR" + matrixType + target + thresholdMode + "Thresholded2.csv"), index = False)
            
            
            df = pandas.read_csv(os.path.join(TABLE_DIR, "Regression", "Finetuned", "YPredictsRFR" + matrixType + target + thresholdMode + "Thresholded2.csv"))
            for threshold in df.columns[1:]:
                YPredict = df[threshold]
                R2[matrixType][target].append(r2_score(Y, YPredict))
                MAE[matrixType][target].append(mean_absolute_error(Y, YPredict))  
        
    print(R2)
    print(MAE)
    
    
    fig, axs = plt.subplots(2, 2, figsize = (16, 16), dpi = 100, sharey = "row")
    for j in range(len(matrixTypes)):
        matrixType = matrixTypes[j]
        axs[0, j].plot(thresholds, R2[matrixType][targets[0]], label = targets[0])
        axs[0, j].plot(thresholds, R2[matrixType][targets[1]], label = targets[1])
        axs[0, j].set_title("R2 Plot(" + matrixType + ")", fontsize = 12)
        axs[0, j].set_xlabel("Threshold")
        axs[0, j].grid(True)
        axs[0, j].legend()
        
        axs[1, j].plot(thresholds, MAE[matrixType][targets[0]], label = targets[0])
        axs[1, j].plot(thresholds, MAE[matrixType][targets[1]], label = targets[1])
        axs[1, j].set_title("MAE Plot(" + matrixType + ")", fontsize = 12)
        axs[1, j].set_xlabel("Threshold")
        axs[1, j].grid(True)
        axs[1, j].legend()
    fig.savefig(os.path.join(FIGURE_DIR, "Regression", "Finetuned", "RFR" + thresholdMode + "Thresholded2.png"))
    '''
    
    optimalThresholds = {"AQ" : {"Bivariate" : 0.33, "Semipartial" : 0.33}, "TX" : {"Bivariate" : 0.31, "Semipartial" : 0.35}}
    errorSq, errorAbs = {m : {t : [] for t in targets} for m in matrixTypes}, {m : {t : [] for t in targets} for m in matrixTypes}
    errorSq["Concatenated"], errorAbs["Concatenated"] = {t : [] for t in targets}, {t : [] for t in targets}
    for i in range(len(targets)):
        target = targets[i]
        Y = None
        for j in range(len(matrixTypes)):
            matrixType = matrixTypes[j]
            filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
            filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
            IDs, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
            if target == "TX":
                Y *= 100.0
            df = pandas.read_csv(os.path.join(TABLE_DIR, "Regression", "Finetuned", "YPredictsRFR" + matrixType + target + thresholdMode + "Thresholded2.csv"))
            threshold = str(optimalThresholds[target][matrixType])
            YPredict = df[threshold].values
            print(target + "  " + matrixType + "  R2 = " + str(r2_score(Y, YPredict)) + "  MAE = " + str(mean_absolute_error(Y, YPredict)))
            errorSq[matrixType][target] = np.power(Y - YPredict, 2)
            errorAbs[matrixType][target] = np.abs(Y - YPredict)
        df = pandas.read_csv(os.path.join(TABLE_DIR, "Regression", "Finetuned", "YPredictsRFRConcatenated" + target + ".csv"))
        YPredict = df["prediction"].values
        print(target + "  Concatenated  R2 = " + str(r2_score(Y, YPredict)) + "  MAE = " + str(mean_absolute_error(Y, YPredict)))
        errorSq["Concatenated"][target] = np.power(Y - YPredict, 2)
        errorAbs["Concatenated"][target] = np.abs(Y - YPredict)
        print(IDs[errorAbs["Concatenated"][target].argmax()] + "  " + str(errorAbs["Concatenated"][target].max()))
    fig, axs = plt.subplots(2, 1, sharex = "col", sharey = "col", figsize = (12, 14), dpi = 100)
    keys = ["Bivariate", "Semipartial", "Concatenated"]
    for i in range(len(targets)):
        target = targets[i]
        axs[i].hist([errorAbs[key][target] for key in keys], label = keys)
        axs[i].set_title("Absolute Error Histogram (" + target + ")")
        axs[i].legend()
        axs[i].set_xlabel("Abs. error")
    # fig.savefig(os.path.join(FIGURE_DIR, "Regression", "Finetuned", "RFRErrorHitogram.png"))

if __name__ == "__main__":
    thresholds = [np.round(i * 0.05, 2) for i in range(10)]
    # index = int(sys.argv[1])
    index = 0
    # runRegressionExperiments(thresholds[index])
    # runConcatenatedRegressionExperiments()
    postjobAnalysis()
    