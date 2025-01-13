import ML.cross_validate as CV
import ML.grid_range_search as GRS
import warnings
from sklearn.exceptions import ConvergenceWarning
from itertools import combinations, product

def run_grid_range_search_experiments(data0, Y, model, processes, include_RS):
    output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO")
    if model == "SVM":
        output_dir_SVM = os.path.join(output_dir, "SVMParameterSearch", "RS" + str(include_RS))
        pathlib.Path(output_dir_SVM).mkdir(parents = True, exist_ok = True)
        GRS.SVM(data0, Y, processes, output_dir_SVM)
    elif model == "RF":
        output_dir_RF = os.path.join(output_dir, "RFParameterSearch", "RS" + str(include_RS))
        pathlib.Path(output_dir_RF).mkdir(parents = True, exist_ok = True)
        GRS.RF(data0, Y, processes, output_dir_RF)
    
def run_RS_feature_ranking_experiments(X, Y, model, features, processes):
    mode = "Classification"
    K = [1, 50, 100, 150, 200, 250, 300, 350, 400, None]
    K = [1, None]
    df_ranking_RS = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_v7", "Transformed", "Statistics", "RS", "RankingsByReliefFSURF.csv"))
    output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", mode, "Untuned", "LOO", "FeatureRankingExperiments")
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    for k in K:
        if k:
            features_RS_ranked = [eval(df_ranking_RS[str(K[0])][i])[0] for i in range(df_ranking_RS.shape[0])]
            M = CV.runSingleFeatureTypeLOOExperiments(X, Y, features, features_RS_ranked, model, mode, processes)
            dfOut = pandas.DataFrame(M)
            dfOut.to_csv(os.path.join(output_dir, model + "_ReliefF(" + str(k) + ")_Predictions.csv"), index = False)
        else:
            features_RS_ranked = [eval(df_ranking_RS["SURF"][i])[0] for i in range(df_ranking_RS.shape[0])]
            M = CV.runSingleFeatureTypeLOOExperiments(X, Y, features, features_RS_ranked, model, mode, processes)
            dfOut = pandas.DataFrame(M)
            dfOut.to_csv(os.path.join(output_dir, model + "_SURF_Predictions.csv"), index = False)
    

def runExperiments(data, features, model, processes, include_RS):
    print("Model = " + model + ", processes = " + str(processes))
    
    
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
    
    statsMethod = "Correlation"
    rankingColumn = "Global"
    # rankingColumn = "LOO Median"
    features_RS = features[FEATURE_TYPE_TO_FEATURES["RS"]]
    df_ranking_RS = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_v7", "Transformed", "Statistics", "RS", "RankingsByAbsolute" + re.sub(" ", "", statsMethod) + ".csv"))
    features_RS_ranked = [eval(df_ranking_RS[rankingColumn][i])[0] for i in range(df_ranking_RS.shape[0])]
    
    
    if include_RS == "All":
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations]
    elif include_RS == "0":
        # all other feature types
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations if not "RS" in combo]
        warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)
    else:
        # target and others, K RS features fixed
        K = int(include_RS)
        features_RS_TopK = features_RS_ranked[:K]
        indices = [i for i in range(len(features_RS)) if features_RS[i] in features_RS_TopK]
        data0["RS"] = data0["RS"][:, indices]
        featureTypeCombinations2 = [combo for combo in featureTypeCombinations if "RS" in combo]
    
    # run ML experiments to find a good cutoff
    run_RS_feature_ranking_experiments(data["RS"], Y, model, features_RS, processes)
    
    # search for good range of parameters
    # run_grid_range_search_experiments(data0, Y, model, processes, include_RS)
    
    # feature type combination, no fine tuning
    '''
    output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", mode, "Untuned", "LOO", "FeatureRankingExperiments")
    pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
    M = runFeatureTypeCombinationLOOExperiments(data0, Y, features, featureTypeCombinations2, model)
    dfOut = pandas.DataFrame(M)
    dfOut.to_csv(os.path.join(output_dir, model + "_Top" + str(K) + "RS_Predictions.csv"), index = False)
    '''
    
    # finetuned
    '''
    if include_RS == "0":
        # others
        cacheDir = os.path.join(PATH_TO_CACHE, mode, "Finetuned", "LOO", model + "FeatureTypeCombinationExperiments", "NoRS")
        pathlib.Path(cacheDir).mkdir(parents = True, exist_ok = True)
        output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Finetuned", "LOO", model)
        pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
        M, CVScoreGrids, CVPredictionGrids = runFeatureTypeCombinationLOOFinetuningExperiments(data0, Y, features, featureTypeCombinations2, model, mode, cacheDir, processes)
        if len(M["Feature Combination"]) == len(featureTypeCombinations):
            dfOut = pandas.DataFrame(M)
            dfOut.to_csv(os.path.join(output_dir, model + "_NoRS_Predictions.csv"), index = False)
            pathlib.Path(cacheDir).mkdir(parents = True, exist_ok = True)
            with open(os.path.join(cacheDir, model + "_NoRS_CVScoreGrids.pkl"), "wb") as outputFile:
                pickle.dump(CVScoreGrids, outputFile, pickle.DEFAULT_PROTOCOL)
            with open(os.path.join(cacheDir, model + "_NoRS_CVPredictionGrids.pkl"), "wb") as outputFile:
                pickle.dump(CVPredictionGrids, outputFile, pickle.DEFAULT_PROTOCOL)
        
    elif not include_RS is None:
        # target + others
        cacheDir = os.path.join(PATH_TO_CACHE, mode, "Finetuned", "LOO", model + "FeatureTypeCombinationExperiments", "RSAndOthers")
        pathlib.Path(cacheDir).mkdir(parents = True, exist_ok = True)
        output_dir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Finetuned", "LOO", model)
        pathlib.Path(output_dir).mkdir(parents = True, exist_ok = True)
        M, CVScoreGrids, CVPredictionGrids = runFeatureTypeCombinationLOOFinetuningExperiments(data0, Y, features, featureTypeCombinations2, model, mode, cacheDir, processes)
        dfOut = pandas.DataFrame(M)
        dfOut.to_csv(os.path.join(output_dir, model + "_RSTop" + str(K) + "_Predictions.csv"), index = False)
        pathlib.Path(cacheDir).mkdir(parents = True, exist_ok = True)
        with open(os.path.join(cacheDir, model + "_RSTop" + str(K) + "_CVScoreGrids.pkl"), "wb") as outputFile:
            pickle.dump(CVScoreGrids, outputFile, pickle.DEFAULT_PROTOCOL)
        with open(os.path.join(cacheDir, model + "_RSTop" + str(K) + "_CVPredictionGrids.pkl"), "wb") as outputFile:
            pickle.dump(CVPredictionGrids, outputFile, pickle.DEFAULT_PROTOCOL)
    '''
        

if __name__ == "__main__":
    model = "AdaBoost"
    processes = 1
    include_RS = "0"
    if "-m" in sys.argv:
        model = sys.argv[sys.argv.index("-m") + 1]
    if "-p" in sys.argv:
        processes = int(sys.argv[sys.argv.index("-p") + 1])
    if "-rs" in sys.argv:
        include_RS = sys.argv[sys.argv.index("-rs") + 1]
    runExperiments(model, processes, include_RS)
    # temp()