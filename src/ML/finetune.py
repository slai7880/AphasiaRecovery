from ML.common import *

def makePredictionFinetuned(X, Y, train, test, model, processes = 1):
    XTrain, XTest, YTrain = X[train], X[test], Y[train]
    grid, estimator = {}, None
    if model == "SVM":
        grid = {"SVM__C" : [1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
                "SVM__gamma" : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]}
        estimator = Pipeline(steps = [("scaler", preprocessing.MinMaxScaler()), ("SVM", SVC(max_iter = 1e5))])
    elif model == "RF":
        grid = {"RF__max_depth" : [4, 8, 12, 16, 20],
                "RF__max_features" : ["log2", "sqrt", 0.1, 0.2, 0.3]}
        estimator = Pipeline(steps = [("scaler", preprocessing.MinMaxScaler()), ("RF", RFC(n_estimators = 800, random_state = 4))])
    elif model == "AdaBoost":
        grid = {"n_estimators" : [400, 450, 500, 550, 600, 650],
                "learning_rate" : [0.4, 0.5]}
        estimator = ABC(base_estimator = DTC(max_depth = 1), random_state = 4)
    NFolds = 5
    # splitter = StratifiedKFold(n_splits = NFolds, shuffle = True, random_state = 4)
    splitter = LeaveOneOut()
    splits = splitter.split([i for i in range(len(YTrain))], YTrain)
    
    clf = CustomizedGridSearchCV(estimator = estimator, param_grid = grid, scoring = f1_score, cv = splits, refit = True, n_jobs = processes)
    '''
    if model == "SVM":
        clf = SVC(gamma = "auto")
    else:
        clf = RFC(n_estimators = 400, max_depth = 20, min_samples_split = 2, min_samples_leaf = 2)
    '''
    
    YPredict = runTrainTest((XTrain, YTrain, XTest, clf))
    
    '''
    predictions = {}
    for C in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        predictions[C] = runTrainTest((XTrain, YTrain, XTest, SVC(C = C, gamma = "scale")))
    '''

    return YPredict, clf

    

def computeCVScores(args):
    seed, X, Y, X_RS, addRS, model, comboIndex, accuracyScores, f1Scores = args
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    YTest, YPredict = np.array([]), np.array([])
    counter = np.array([0] * X_RS.shape[1])
    for train, test in skf.split([j for j in range(len(Y))], Y):
        YTest = np.concatenate((YTest, Y[test]))
        YPredictCurrent, _ = makePredictionFinetuned(X, Y, X_RS, train, test, addRS, model)
        YPredict = np.concatenate((YPredict, YPredictCurrent))
    accuracy, f1 = accuracy_score(YTest, YPredict), f1_score(YTest, YPredict)
    # accuracy, f1 = 0, 0
    return accuracy, f1, counter

def runMultiCVFinetuningExperiments(data0, Y, features, featureTypeCombinations, model, K = 100, processes = 1):
    seeds = [i for i in range(K)]
    data = {key : data0[key].copy() for key in data0}
    if model == "SVM":
        data["DM"] = (data["DM"] - data["DM"].min(axis = 0)) / (data["DM"].max(axis = 0) - data["DM"].min(axis = 0))
        data["LS"] = (data["LS"] - data["LS"].min(axis = 0)) / (data["LS"].max(axis = 0) - data["LS"].min(axis = 0))
        data["RS"] = (data["RS"] - data["RS"].min(axis = 0)) / (data["RS"].max(axis = 0) - data["RS"].min(axis = 0))
    accuracyScores, f1Scores = {seed : [0.0] * len(featureTypeCombinations) for seed in seeds}, {seed : [0.0] * len(featureTypeCombinations) for seed in seeds}
    
    for i in range(len(featureTypeCombinations)):
        X = [np.zeros((len(Y), 1))] + [data[f] for f in featureTypeCombinations[i] if not f == "RS"]
        X = np.hstack(X)
        args = []
        for seed in seeds:
            args.append((seed, X, Y, data["RS"], "RS" in featureTypeCombinations[i], 0.31, model, i, accuracyScores, f1Scores))
        if processes == 1:
            results = []
            for j in range(len(args)):
                accuracy, f1, counter = computeCVScores(args[j])
                results.append((accuracy, f1, counter))
        else:
            pool = Pool(processes)
            results = pool.map(computeCVScores, args)
        for j in range(len(seeds)):
            seed = seeds[j]
            accuracy, f1, _ = results[j]
            accuracyScores[seed][i] = accuracy
            f1Scores[seed][i] = f1
        if "RS" in featureTypeCombinations[i] and len(featureTypeCombinations[i]) == 1:
            RSFeatureCounter = np.array([0] * data["RS"].shape[1])
            for j in range(len(seeds)):
                _, _, counter = results[j]
                RSFeatureCounter += counter
            np.save("RSFeatureCounter.npy", RSFeatureCounter)
        '''
        print("Features " + str(featureTypeCombinations[i]))
        print("Accuracy = " + str(accuracy))
        print("F1 = " + str(f1))
        print()
        '''
    outputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", "Classification", "Untuned", "MultiCV")
    pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
    name = model
    
    dfAccuracy = pandas.DataFrame(accuracyScores)
    dfAccuracy["Feature Combination"] = featureTypeCombinations
    dfAccuracy["Mean"] = np.mean(dfAccuracy[seeds].values, axis = 1)
    dfAccuracy["Variance"] = np.var(dfAccuracy[seeds].values, axis = 1)
    dfAccuracy = dfAccuracy[["Feature Combination"] + seeds + ["Mean", "Variance"]].round(4)
    dfAccuracy.to_csv(os.path.join(outputDir, name + "AccuracyScores_RSThresholdedByTXPerFold_NewNorm.csv"), index = False)
    
    dfF1 = pandas.DataFrame(f1Scores)
    dfF1["Feature Combination"] = featureTypeCombinations
    dfF1["Mean"] = np.mean(dfF1[seeds].values, axis = 1)
    dfF1["Variance"] = np.var(dfF1[seeds].values, axis = 1)
    dfF1 = dfF1[["Feature Combination"] + seeds + ["Mean", "Variance"]].round(4)
    dfF1.to_csv(os.path.join(outputDir, name + "F1Scores_RSThresholdedByTXPerFold_NewNorm.csv"), index = False)

def runFeatureTypeCombinationLOOFinetuningExperiments(data0, Y, featureTypeCombinations, model, mode, cacheDir, processes, stepSize = None):
    data = {key : data0[key].copy() for key in data0}
    M = {"Feature Combination" : []}
    for i in range(len(Y)):
        M[i] = []
    CVScoreGrids, CVPredictionGrids = {}, {}
    start = 0
    '''
    try:
        with open(os.path.join(cacheDir, "PredictionsTemp.pkl"), "rb") as inputFile:
            M = pickle.load(inputFile)
        with open(os.path.join(cacheDir, "CVScoreGridsTemp.pkl"), "rb") as inputFile:
            CVScoreGrids = pickle.load(inputFile)
        with open(os.path.join(cacheDir, "CVPredictionGridsTemp.pkl"), "rb") as inputFile:
            CVPredictionGrids = pickle.load(inputFile)
        start = len(M["Feature Combination"])
        progressStr = str(len(M["Feature Combination"])) + " / " + str(len(featureTypeCombinations))
        print("Check point found. Last feature combination was " + str(M["Feature Combination"][-1]) +  " (" + progressStr + ").")
        print("Now starting from " + str(featureTypeCombinations[start]) + ".")
    except:
        print("No checkpoint files found.")
    '''
    end = len(featureTypeCombinations)
    if stepSize:
        end = start + stepSize
    for i in range(end - 1, start, -1):
        print("Running finetuning experiments on feature type combination " + str(featureTypeCombinations[i]) + " (" + str(i + 1) + " / " + str(len(featureTypeCombinations)) + ").")
        M["Feature Combination"].append(featureTypeCombinations[i])
        X = [data[f] for f in featureTypeCombinations[i]]
        X = np.hstack(X)
        print("X.shape = " + str(X.shape) + "  Y.shape = " + str(Y.shape))
        YTest = Y
        CVScoreGrids[featureTypeCombinations[i]] = [None] * len(X)
        CVPredictionGrids[featureTypeCombinations[i]] = [None] * len(X)
        timeStart = time.time()
        for test in trange(len(X)):
            train = [j for j in range(len(X)) if j != test]
            YPredict, clf = makePredictionFinetuned(X, Y, train, [test], model, processes)
            M[test].append(YPredict[0])
            CVScoreGrids[featureTypeCombinations[i]][test] = clf.CVScoreGrid
            CVPredictionGrids[featureTypeCombinations[i]][test] = clf.CVPredictionGrid
        timeEnd = time.time()
        timeElapsed = np.round((timeEnd - timeStart) / 60, 2)
        print("Run time = " + str(timeElapsed) + " minutes.")
        print("Saving progress.\n")
        with open(os.path.join(cacheDir, "PredictionsTemp.pkl"), "wb") as outputFile:
            pickle.dump(M, outputFile, pickle.DEFAULT_PROTOCOL)
        with open(os.path.join(cacheDir, "CVScoreGridsTemp.pkl"), "wb") as outputFile:
            pickle.dump(CVScoreGrids, outputFile, pickle.DEFAULT_PROTOCOL)
        with open(os.path.join(cacheDir, "CVPredictionGridsTemp.pkl"), "wb") as outputFile:
            pickle.dump(CVPredictionGrids, outputFile, pickle.DEFAULT_PROTOCOL)
    
    return M, CVScoreGrids, CVPredictionGrids