from common import *
from ML import cross_validate as CV
from ML import grid_range_search as GRS
from ML import finetune as FT
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA, FastICA

warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)

def run_grid_range_search_experiments(X, Y, model, processes, output_dir):
    if model == "SVM":
        GRS.SVM(X, Y, processes, output_dir)
    elif model == "RF":
        GRS.RF(X, Y, processes, output_dir)
    elif model == "AdaBoost":
        GRS.AdaBoost(X, Y, processes, output_dir)


def run_RS_feature_ranking_experiments(X, Y, model, features, processes):
    mode = "Classification"
    K = [1, 50, 100, 150, 200, 250, 300, 350, 400, None]
    K = [None, 400, 350, 300, 250, 200, 150, 100]
    df_ranking_RS = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_v7", "Transformed", "Statistics", "RS", "RankingsByReliefFSURF.csv"))
    output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", mode, "Untuned", "LOO", "FeatureRankingExperiments", model)
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    for k in K:
        if k:
            features_RS_ranked = [eval(df_ranking_RS[str(k)][i])[0] for i in range(df_ranking_RS.shape[0])]
            M = CV.runSingleFeatureTypeLOOExperiments(X, Y, features, features_RS_ranked, model, mode, processes)
            dfOut = pandas.DataFrame(M)
            dfOut.to_csv(os.path.join(output_dir, model + "_ReliefF(" + str(k) + ")_Predictions.csv"), index = False)
        else:
            features_RS_ranked = [eval(df_ranking_RS["SURF"][i])[0] for i in range(df_ranking_RS.shape[0])]
            M = CV.runSingleFeatureTypeLOOExperiments(X, Y, features, features_RS_ranked, model, mode, processes)
            dfOut = pandas.DataFrame(M)
            dfOut.to_csv(os.path.join(output_dir, model + "_SURF_Predictions.csv"), index = False)

def run_finetuning_experiments(data0, Y, model, featureTypeCombinations2, processes, stepSize):
    cacheDir = os.path.join(PATH_TO_CACHE, "Classification", "Finetuned", "LOO", model + "FeatureTypeCombinationExperiments", "FA_PS_RS_reduced")
    pathlib.Path(cacheDir).mkdir(parents = True, exist_ok = True)
    output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Finetuned", "LOO", model)
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    M, CVScoreGrids, CVPredictionGrids = FT.runFeatureTypeCombinationLOOFinetuningExperiments(data0, Y, featureTypeCombinations2, model, "Classification", cacheDir, processes, stepSize)
    dfOut = pandas.DataFrame(M)
    dfOut.to_csv(os.path.join(output_dir, model + "_FA_PS_RS_reduced_Predictions.csv"), index = False)
    pathlib.Path(cacheDir).mkdir(parents = True, exist_ok = True)
    with open(os.path.join(cacheDir, model + "_FA_PS_RS_reduced_CVScoreGrids.pkl"), "wb") as outputFile:
        pickle.dump(CVScoreGrids, outputFile, pickle.DEFAULT_PROTOCOL)
    with open(os.path.join(cacheDir, model + "_FA_PS_RS_reduced_CVPredictionGrids.pkl"), "wb") as outputFile:
        pickle.dump(CVPredictionGrids, outputFile, pickle.DEFAULT_PROTOCOL)
    
def filterFeatures(data0, features, featureTypeCombinations, target, statsMethod, rankingColumn, K):
    features_target = features[target]
    df_ranking = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Statistics", target, "RankingsByAbsolute" + re.sub(" ", "", statsMethod) + ".csv"))
    features_target_ranked = [eval(df_ranking[rankingColumn][i])[0] for i in range(df_ranking.shape[0])]
    
    data2 = {k : data0[k].copy() for k in data0}
    featureTypeCombinations2 = []
    if K == "All":
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations]
    elif K == "0":
        data2.pop(target)
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations if not target in combo]
    else:
        # target and others, K RS features fixed
        features_target_TopK = features_target_ranked[:int(K)]
        indices = [i for i in range(len(features_target)) if features_target[i] in features_target_TopK]
        data2[target] = data2[target][:, indices]
        # featureTypeCombinations2 = [combo for combo in featureTypeCombinations if target in combo]
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations]
    return data2, featureTypeCombinations2

def runExperiments(data, features, model, processes, stepSize):
    print("Model = " + model + ", processes = " + str(processes))
    
    featureTypes = ["AQ", "CS", "DM", "LS", "PS_W", "FA", "PS_G", "RS"]
    
    featureTypeCombinations = []
    for i in range(1, len(featureTypes) + 1):
        featureTypeCombinations += list(combinations(featureTypes, i))
    
    Y = data["TX_0.25"]
    counts = {}
    for y in Y:
        counts[y] = 1 + counts.get(y, 0)
    print("Label distribution : " + str(counts))
    
    mode = "Classification"
    
    statsMethod = "Correlation"
    rankingColumn = "Global"
    # rankingColumn = "LOO Median"
    
    Ks = getFeatureTypeK()
    for target in Ks:
        data, featureTypeCombinations = filterFeatures(data, features, featureTypeCombinations, target, statsMethod, rankingColumn, Ks[target])
    print("Feature shapes after reduction:")
    for f in featureTypes:
        print(f + "  " + str(data[f].shape))
    print("Number of all feature type combinations: " + str(len(featureTypeCombinations)))
    
    
    # search for good range of parameters
    '''
    for target in ["PS_G", "PS_W", "RS"]:
        for k in trange(1, data[target].shape[1] + 1):
            data2, _ = filterFeatures(data, features, featureTypeCombinations, target, statsMethod, rankingColumn, str(k))
            output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO", model + "ParameterSearch_" + target + "_Only", target + str(k))
            pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
            run_grid_range_search_experiments(data2[target], Y, model, processes, output_dir)
    '''
    # fine-tuning
    run_finetuning_experiments(data, Y, model, featureTypeCombinations, processes, stepSize)

def fineArgMaxLast(L):
    index, M = 0, -sys.maxsize
    for i in range(len(L)):
        if L[i] >= M:
            M = L[i]
            index = i
    return index

def getFeatureTypeK():
    featureTypesToReduce = ["FA", "PS_G", "PS_W", "RS"]
    Ks = {}
    for target in featureTypesToReduce:
        dfRF = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO", "RFParameterSearch_" + target + "_Only", "F1ScoresOptimal.csv"))
        dfSVM = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO", "SVMParameterSearch_" + target + "_Only", "F1ScoresOptimal.csv"))
        Ks[target] = str(max(fineArgMaxLast(dfRF["gini"]), fineArgMaxLast(dfSVM["RBF"])) + 1)
    return Ks

def run_single_featureset_finetuning_experiments():
    data, features = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = False)
    df = pandas.read_excel(os.path.join(CROSS_SITE_DATASET, "aphasia_syndrome_data_v7.xlsx"))
    syndromes = df["Syndrome"].values.reshape(-1, 1)
    syndromes_unique = np.array(sorted(list(set(df["Syndrome"].values.tolist())))).reshape(-1, 1)
    encoder = OneHotEncoder(sparse = False)
    encoder.fit(syndromes_unique)
    X = encoder.transform(syndromes)
    Y = data["TX_0.25"]
    for model in ["SVM", "RF"]:
        YPredict = []
        for test in trange(len(X)):
            train = [j for j in range(len(X)) if j != test]
            temp, _ = FT.makePredictionFinetuned(X, Y, train, [test], model, processes = 10)
            YPredict.append(temp[0])
        TN, N = 0, 0
        for i in range(len(Y)):
            TN += Y[i] == 0 and YPredict[i] == 0
            N += (Y[i] == 0).squeeze()
        scores = {"Accuracy" : accuracy_score(Y, YPredict), "F1" : f1_score(Y, YPredict), "Sensitivity" : recall_score(Y, YPredict), "Selectivity" : TN / N}
        print(model + " YTrue = " + str(Y))
        print(model + " YPredict = " + str(YPredict))
        print(model + " scores = " + str(scores))
    
def run_AQ_prediction():
    data, features = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = False)
    Y = data["AQ"].squeeze()
    X = data["RS"]
    X = X[:, :200]
    LOO = LeaveOneOut()
    Y_true = []
    Y_predict = []
    for train, test in LOO.split([i for i in range(X.shape[0])], Y):
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        Y_true.append(Y_test)
        model = SVR()
        model.fit(X_train, Y_train)
        Y_predict.append(model.predict(X_test))
    Y_true = np.concatenate(Y_true)
    Y_precict = np.concatenate(Y_predict)
    print(np.linalg.norm(Y_true - Y_predict) / X.shape[0])

def recover_correlation_matrix(features, values, regions, region_indices):
    M = np.matrix([[0.0] * len(regions) for _ in range(len(regions))])
    for i in range(len(features)):
        feature_split = features[i].split("_vs_")
        region1 = feature_split[0]
        region2 = feature_split[1][:len(feature_split[1]) - 3]
        M[region_indices[region1], region_indices[region2]] = values[i]
    return M

def run_CNN_experiments(data, features):
    '''
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    region_indices = {regions[i]: i for i in range(len(regions))}
    '''
    RS_detransformed = np.tanh(data["RS"])
    RS_rescaled = (RS_detransformed - (-1)) / (1 - (-1))
    regions = []
    for i in range(len(features["RS"])):
        feature_split = features["RS"][i].split("_vs_")
        region1 = feature_split[0]
        region2 = feature_split[1][:len(feature_split[1]) - 3]
        regions.append(region1)
        regions.append(region2)
    regions = list(set(regions))
    regions.sort()
    region_indices = {regions[i]: i for i in range(len(regions))}
    tuples = [(i, data["AQ"][i]) for i in range(data["AQ"].shape[0])]
    tuples.sort(key = lambda x : x[1])
    for t in tuples:
        M = recover_correlation_matrix(features["RS"], RS_detransformed[t[0], :], regions, region_indices)
        fig, ax = plt.subplots(figsize = (10, 8))
        sns.heatmap(M, ax = ax)
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    model, processes, stepSize = "SVM", 1, None
    if "-m" in sys.argv:
        model = sys.argv[sys.argv.index("-m") + 1]
    if "-p" in sys.argv:
        processes = int(sys.argv[sys.argv.index("-p") + 1])
    if "-stepsize" in sys.argv:
        stepSize = int(sys.argv[sys.argv.index("-stepsize") + 1])
    data, features = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = False)
    # runExperiments(data, features, model, processes, stepSize)
    
    # run_single_featureset_finetuning_experiments()
    # run_AQ_prediction()
    run_CNN_experiments(data, features)