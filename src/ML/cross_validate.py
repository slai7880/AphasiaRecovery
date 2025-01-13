from ML.common import *
# to do: change model from string to actual class

def runFeatureTypeCombinationLOOExperiments(data0, Y, featureTypes, featureTypeCombinations, model, modelArgs = None, mode = "Classification", processes = 1):
    data = {key : data0[key].copy() for key in data0}
    
    M = {"Feature Combination" : featureTypeCombinations}
    for i in range(len(Y)):
        M[i] = [0] * len(featureTypeCombinations)
    for i in range(len(featureTypeCombinations)):
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
    return M
    
def runSingleFeatureTypeLOOExperiments(X, Y, features, featuresRanked, model, mode, processes):
    K = len(features)
    M = {"K" : [i for i in range(K)]}
    for i in range(len(Y)):
        M[i] = [0] * K
    XOriginal = X
    for k in trange(1, len(featuresRanked) + 1):
        indices = [i for i in range(len(features)) if features[i] in featuresRanked[:k]]
        X = XOriginal[:, indices]
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