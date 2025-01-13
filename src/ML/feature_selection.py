from ML.common import *
from skrebate import ReliefF, SURF

def run_top_k_experiments(data, features, model, processes = 1, targetFeatureType = "RS", statsMethod = "Correlation", rankingColumn = "Global"):
    featuresToReduce = features[FEATURE_TYPE_TO_FEATURES[featureTypeToReduce]]
    dfRanking = pandas.read_csv(os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", "Statistics", featureTypeToReduce, "RankingsByAbsolute" + re.sub(" ", "", statsMethod) + ".csv"))
    featuresRanked = [eval(dfRanking[rankingColumn][i])[0] for i in range(dfRanking.shape[0])]
    
    # target + others
    '''
    k0 = 0
    for K in range(k0, 2):
        outputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Untuned", "LOO", "TopKExperiments", featureTypeToReduce, re.sub(" ", "", statsMethod), re.sub(" ", "", rankingColumn), model)
        pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
        M = runFeatureTypeCombinationLOOExperiments(data0, Y, features, featureTypeCombinations2, model, mode, processes)
        dfOut = pandas.DataFrame(M)
        dfOut.to_csv(os.path.join(outputDir, model + "_Top" + str(K) + "_Predictions.csv"), index = False)
    '''
    
    # target only
    
    outputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Untuned", "LOO", "TopKExperiments", featureTypeToReduce + "_Only", re.sub(" ", "", statsMethod), re.sub(" ", "", rankingColumn), model)
    pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
    M = runSingleFeatureTypeLOOExperiments(data[featureTypeToReduce], Y, features, featuresRanked, model, mode, processes)
    # dfOut = pandas.DataFrame(M)
    # dfOut.to_csv(os.path.join(outputDir, model + "_Predictions.csv"), index = False)

def run_SURF_experiments(X, Y):
    K = [i for i in range(1, 21)]
    M = {}
    for k in K:
        selector = SURF(n_features_to_select = K[0], discrete_threshold = 1)
        selector.fit(X, Y)
        M[k] = selector.feature_importances_
    df = pandas.DataFrame(M)
    