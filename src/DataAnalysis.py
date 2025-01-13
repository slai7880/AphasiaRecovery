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
    

def runRSStabilityTest(X, Y, models, stabilityDir, type):
    pathlib.Path(stabilityDir).mkdir(parents = True, exist_ok = True)
    seedsSKF = [227, 333, 763, 715, 344, 995, 560, 589, 784, 673, 924, 404, 952, 493, 624, 998, 118, 729, 937, 696]
    seedsRF = [312, 726, 10, 11, 119, 366, 385, 205, 203, 607, 334, 973, 721, 221, 724, 97, 911, 644, 499, 12]
    
    if type == 0:
        optimalThresholds = {m : {seed : [] for seed in seedsSKF} for m in models}
        for seed in seedsSKF:
            for test in range(len(X)):
                train = [i for i in range(len(X)) if i != test]
                XTrain, YTrain = X[train], Y[train]
                for model in models:
                    thresholdOptimal, _ = findOptimalThreshold(XTrain, YTrain, model, seedSKF = seed, seedRF = seedsRF[0])
                    optimalThresholds[model][seed].append(thresholdOptimal)
        with open(os.path.join(stabilityDir, "OptimalThresholdsVarSplitsForClassification.pkl"), "wb") as outputFile:
            pickle.dump(optimalThresholds, outputFile, pickle.DEFAULT_PROTOCOL)
    else:
        optimalThresholds = {seed : [] for seed in seedsRF}
        for seed in seedsRF:
            for test in range(len(X)):
                train = [i for i in range(len(X)) if i != test]
                XTrain, YTrain = X[train], Y[train]
                thresholdOptimal, _ = findOptimalThreshold(XTrain, YTrain, "RF", seedSKF = seedsSKF[0], seedRF = seed)
                optimalThresholds[seed].append(thresholdOptimal)
        with open(os.path.join(stabilityDir, "OptimalThresholdsVarRFSeedsForClassification.pkl"), "wb") as outputFile:
            pickle.dump(optimalThresholds, outputFile, pickle.DEFAULT_PROTOCOL)
    
def makeDensityPlots(models, optimalThresholds):
    fig, axs = plt.subplots(2, 1, figsize = (10, 8), dpi = 100, sharex = True)
    for i in range(len(models)):
        model = models[i]
        sns.kdeplot(optimalThresholds[model], ax = axs[i])
        axs[i].scatter(optimalThresholds[model], [0] * len(optimalThresholds[model]))
        axs[i].set_title(model + " Optimal Threshold Density")
        axs[i].set_xlabel("Threshold")
    fig.tight_layout()
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments", "Transformed", "Statistics")
    fig.savefig(os.path.join(figDir, "LOOOptimalThresholdsForClassificationDensityPlots.png"))

    regions = None
    file = open(os.path.join(PATH_TO_RS_DATA, "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    
    
    regionPairs = featuresRaw["restingstate_bivariate_correlations"]
    fig, axs = plt.subplots(1, 2, figsize = (32, 24), dpi = 100)
    for i in range(len(models)):
        regionPairCounts = [0] * len(regionPairs)
        for test in range(len(X)):
            train = [j for j in range(len(X)) if j != test]
            XTrain, YTrain = X[train], Y[train]
            for j in range(X.shape[1]):
                corr, _ = sp.stats.pearsonr(XTrain[:, j].squeeze(), YTrain)
                if np.abs(corr) >= optimalThresholds[models[i]][test]:
                    regionPairCounts[j] += 1
        M = np.zeros((len(regions), len(regions)))
        for j in range(len(regionPairCounts)):
            pair = regionPairs[j].split("_vs_")
            pair[1] = pair[1][:len(pair[1]) - 3]
            M[regionIndices[pair[0]], regionIndices[pair[1]]] = regionPairCounts[j]
        M += M.T
        axs[i].imshow(M, interpolation = "nearest", origin = "upper", cmap = plt.cm.Blues, vmin = 0, vmax = 55)
        for j in range(M.shape[0]):
            for k in range(M.shape[1]):
                color = "white"
                if M[j, k] <= 55 / 2:
                    color = "black"
                text = str(M[j, k])
                text = str(int(M[j, k]))
                axs[i].text(k, j, text, ha = "center", va = "center", color = color, fontsize = 10)
        axs[i].set_title("Histogram of Selected RS Features (" + models[i] + ")", fontsize = 32)
        axs[i].set_xticks([i for i in range(len(regions))])
        axs[i].set_yticks([i for i in range(len(regions))])
        axs[i].set_xticklabels(regions, fontsize = 12)
        axs[i].set_yticklabels(regions, fontsize = 12)
        axs[i].xaxis.tick_top()
        plt.setp(axs[i].get_xticklabels(), rotation = 90, ha = "left", rotation_mode = "anchor")
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments", "Transformed", "Statistics")
    fig.savefig(os.path.join(figDir, "RSFeatureHistogramn.png"))
      
def getStatisticsRankings(X, Y, features, statsMethod):
    def getValue(X, Y, statsMethod):
        value = None
        if statsMethod == "Correlation":
            value, _ = sp.stats.pearsonr(X, Y)
        elif statsMethod == "Chi Squared":
            value, _ = chi2(X.reshape(-1, 1), Y)
            value = value[0]
        elif statsMethod == "ANOVA F-value":
            value, _ = f_classif(X.reshape(-1, 1), Y)
            value = value[0]
        else:
            value = mutual_info_classif(X.reshape(-1, 1), Y)
            value = value[0]
        return value
        
    def getRankings(tuples):
        tuples.sort(reverse = True, key = lambda x : x[-1])
        rankings = [(t[0], np.round(t[2], 4)) for t in tuples]
        featuresRanked = [t[0] for t in rankings]
        return rankings, featuresRanked
    
    # global
    tuples = []
    for i in range(len(features)):
        value = getValue(X[:, i].squeeze(), Y, statsMethod)
        tuples.append((features[i], value, np.abs(value)))
    rankingsGlobal, featuresRankedGlobal = getRankings(tuples)
    
    # LOO
    values = np.zeros(X.shape, dtype = "float")
    for test in range(len(X)):
        train = [i for i in range(len(X)) if i != test]
        for i in range(len(features)):
            value = getValue(X[train, i].squeeze(), Y[train], statsMethod)
            values[test, i] = value
    
    # LOO median
    medians = np.median(values, axis = 0)
    tuplesMedian = []
    for i in range(len(features)):
        r = medians[i]
        tuplesMedian.append((features[i], r, np.abs(r)))
    rankingsMedian, featuresRankedMedian = getRankings(tuplesMedian)
    
    # LOO mean
    means = np.median(values, axis = 0)
    tuplesMean = []
    for i in range(len(features)):
        tuplesMean.append((features[i], means[i], np.abs(means[i])))
    rankingsMean, featuresRankedMean = getRankings(tuplesMean)
    
    rankings = {"Global" : rankingsGlobal, "LOO Median" : rankingsMedian, "LOO Mean" : rankingsMean}
    featuresRanked = {"Global" : featuresRankedGlobal, "LOO Median" : featuresRankedMedian, "LOO Mean" : featuresRankedMean}
    
    return rankings, featuresRanked

# only works if no duplicate in L1 and in L2
def getIntersectionSizes(L1, L2, K):
    assert len(set(L1)) == len(L1) and len(set(L2)) == len(L2)
    assert len(L1) <= K and len(L2) <= K
    counts = [0] * K
    for k in range(K):
        counts[k] = int(L1[k] == L2[k]) 
        if k > 0:
            counts[k] += counts[k - 1] + int(L1[k] in L2[:k]) + int(L2[k] in L1[:k])
    return counts

def makeStatisticsRankings(X, Y, features, statsMethods, tableDir = None, figDir = None):
    rankings, featuresRanked = {}, {}
    for i in range(len(statsMethods)):
        method = statsMethods[i]
        rankings[method], featuresRanked[method] = getStatisticsRankings(X, Y, features, method)
        dfOut = pandas.DataFrame(rankings[method])
        if tableDir:
            dfOut.to_csv(os.path.join(tableDir, "RankingsByAbsolute" + re.sub(" ", "", method) + ".csv"), index = False)
    
    if figDir:
        # global-median intersection counts for each method
        fig, axs = plt.subplots(2, 2, figsize = (20, 20), dpi = 100)
        for i in range(len(statsMethods)):
            featuresRankedGlobal, featuresRankedMedian = featuresRanked[statsMethods[i]]["Global"], featuresRanked[statsMethods[i]]["LOO Median"]
            intersectionSizes = getIntersectionSizes(featuresRankedGlobal, featuresRankedMedian, len(featuresRankedMedian))
            tau, _ = sp.stats.kendalltau(featuresRankedGlobal, featuresRankedMedian)
            axs[i // 2, i % 2].plot(intersectionSizes)
            axs[i // 2, i % 2].set_title(statsMethods[i] + " (tau = " + str(np.round(tau, 4)) + ")", fontsize = 16)
            axs[i // 2, i % 2].set_xlabel("K", fontsize = 14)
            axs[i // 2, i % 2].set_ylabel("Global Values and LOO Medians Intersection Size", fontsize = 14)
            axs[i // 2, i % 2].grid(True)
        fig.savefig(os.path.join(figDir, "Global_vs_LOOMedian_IntersectionSizes.png"))
        plt.close(fig)
        
        # global intersection counts between methods
        fig, axs = plt.subplots(2, 3, figsize = (32, 20), dpi = 100)
        count = 0
        for i in range(len(statsMethods)):
            for j in range(i + 1, len(statsMethods)):
                featuresRankedGlobal1, featuresRankedGlobal2 = featuresRanked[statsMethods[i]]["Global"], featuresRanked[statsMethods[j]]["Global"]
                intersectionSizes = getIntersectionSizes(featuresRankedGlobal1, featuresRankedGlobal2, len(featuresRankedGlobal1))
                tau, _ = sp.stats.kendalltau(featuresRankedGlobal1, featuresRankedGlobal2)
                axs[count // 3, count % 3].plot(intersectionSizes)
                axs[count // 3, count % 3].set_title(statsMethods[i] + " vs " + statsMethods[j] + " (tau = " + str(np.round(tau, 4)) + ")", fontsize = 16)
                axs[count // 3, count % 3].set_xlabel("K", fontsize = 14)
                axs[count // 3, count % 3].set_ylabel("Intersection Size", fontsize = 14)
                axs[count // 3, count % 3].grid(True)
                count += 1
        fig.savefig(os.path.join(figDir, "Global_BetweenMethods_IntersectionSizes.png"))
        plt.close(fig)
        
        # median intersection counts between methods
        fig, axs = plt.subplots(2, 3, figsize = (32, 20), dpi = 100)
        count = 0
        for i in range(len(statsMethods)):
            for j in range(i + 1, len(statsMethods)):
                featuresRankedGlobal1, featuresRankedGlobal2 = featuresRanked[statsMethods[i]]["LOO Median"], featuresRanked[statsMethods[j]]["LOO Median"]
                intersectionSizes = getIntersectionSizes(featuresRankedGlobal1, featuresRankedGlobal2, len(featuresRankedGlobal1))
                tau, _ = sp.stats.kendalltau(featuresRankedGlobal1, featuresRankedGlobal2)
                axs[count // 3, count % 3].plot(intersectionSizes)
                axs[count // 3, count % 3].set_title(statsMethods[i] + " vs " + statsMethods[j] + " (tau = " + str(np.round(tau, 4)) + ")", fontsize = 16)
                axs[count // 3, count % 3].set_xlabel("K", fontsize = 14)
                axs[count // 3, count % 3].set_ylabel("Intersection Size", fontsize = 14)
                axs[count // 3, count % 3].grid(True)
                count += 1
        fig.savefig(os.path.join(figDir, "LOOMedian_BetweenMethods_IntersectionSizes.png"))
        plt.close(fig)
    
def makeSURFRankings(X, Y, features, tableDir, figDir):
    K = [1, 50, 100, 150, 200, 250, 300, 350, 400, None]
    M = {}
    for k in K:
        if k:
            selector = ReliefF(n_features_to_select = 2, n_neighbors = k, discrete_threshold = 1)
            selector.fit(X, Y)
            tuples = [(features[i], selector.feature_importances_[i]) for i in range(len(features))]
            tuples.sort(reverse = True, key = lambda x : x[1])
            for i in range(len(tuples)):
                tuples[i] = (tuples[i][0], np.round(tuples[i][1], 4))
            M[k] = tuples
        else:
            selector = SURF(n_features_to_select = 2, discrete_threshold = 1)
            selector.fit(X, Y)
            tuples = [(features[i], selector.feature_importances_[i]) for i in range(len(features))]
            tuples.sort(reverse = True, key = lambda x : x[1])
            for i in range(len(tuples)):
                tuples[i] = (tuples[i][0], np.round(tuples[i][1], 4))
            M["SURF"] = tuples
    dfOut = pandas.DataFrame(M)
    dfOut.to_csv(os.path.join(tableDir, "RankingsByReliefFSURF.csv"), index = False)

def RSStatisticsAnalysis():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    X_RS = data["RS"]
    Y = data["TX_0.25"]
    featuresRS = featuresRaw["restingstate_bivariate_correlations"]
    statsMethods = ["Correlation", "Chi Squared", "ANOVA F-value", "Mutual Information"]
    
    tableDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Statistics", "RS")
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Statistics", "RS")
    for dir in (tableDir, figDir):
        if dir:
            pathlib.Path(dir).mkdir(parents = True, exist_ok = True)
    # makeStatisticsRankings(X_RS, Y, featuresRS, statsMethods, tableDir, figDir)
    makeSURFRankings(X_RS, Y, featuresRS, tableDir, figDir)

def PSStatisticsAnalysis():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    X = {"PS_G" : data["PS_G"], "PS_W" : data["PS_W"]}
    Y = data["TX_0.25"]
    features = {"PS_G" : featuresRaw["percent_spared_in_gray_matter"], "PS_W" : featuresRaw["percent_spared_in_white_matter"]}
    
   
    
    statsMethods = ["Correlation", "Chi Squared", "ANOVA F-value", "Mutual Information"]
    for PS in ("PS_G", "PS_W"):
        tableDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", "Statistics", PS)
        figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments", "Transformed", "Statistics", PS)
        
        # only need the meaningful columns
        indices = []
        for i in range(X[PS].shape[1]):
            if len(set(X[PS][:, i].tolist())) > 2:
                indices.append(i)
        X[PS] = X[PS][:, indices]
        features[PS] = [features[PS][i] for i in indices]
        makeStatisticsRankings(X[PS], Y, features[PS], statsMethods, tableDir, figDir)

def evaluateTopKExperimentsHelper(K, YTrue, targetFeatureType, models, metrics, statsMethod, mode, rankingColumns, targetOnly, figDir):
    optimalScores = [None for _ in range(len(models))]
    for iRankingColumn in range(len(rankingColumns)):
        rankingColumn = rankingColumns[iRankingColumn]
        pathlib.Path(os.path.join(figDir, re.sub(" ", "", rankingColumn))).mkdir(parents = True, exist_ok = True)
        optimalScores[iRankingColumn] = {model : {metric : [] for metric in metrics}  for model in models}
        if targetOnly:
            for model in models:
                inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Untuned", "LOO", "TopKExperiments", targetFeatureType + "_Only", re.sub(" ", "", statsMethod), re.sub(" ", "", rankingColumn), model)
                dfPredictions = pandas.read_csv(os.path.join(inputDir, model + "_Predictions.csv"))
                scores = {"G : 0; P : 0" : [], "G : 0; P : 1" : [], "G : 1; P : 0" : [], "G : 1; P : 1" : [], "Accuracy" : [], "F1" : [], "Precision" : [], "Recall" : []}
                for i in range(dfPredictions.shape[0]):
                    YPredict = dfPredictions.iloc[i, 1:].values.astype(int)
                    cm = confusion_matrix(YTrue, YPredict)
                    for j in range(cm.shape[0]):
                        for k in range(cm.shape[1]):
                            scores["G : " + str(j) + "; P : " + str(k)].append(cm[j, k])
                    scores["Accuracy"].append(accuracy_score(YTrue, YPredict))
                    scores["F1"].append(f1_score(YTrue, YPredict))
                    scores["Precision"].append(precision_score(YTrue, YPredict))
                    scores["Recall"].append(recall_score(YTrue, YPredict))
                for metric in metrics:
                    optimalScores[iRankingColumn][model][metric] = scores[metric]
            if K < len(optimalScores[iRankingColumn][model][metric]):
                fig, axs = plt.subplots(2, 2, sharey = True, figsize = (40, 20), dpi = 100)
                for i in range(len(metrics)):
                    metric = metrics[i]
                    for j in range(len(models)):
                        model = models[j]
                        axs[i // 2, i % 2].plot([k for k in range(1, K + 1)], optimalScores[iRankingColumn][model][metric][:K], label = model)
                    axs[i // 2, i % 2].set_title("Optimal " + metric + " Scores over Top-K " + targetFeatureType + "-Only Features", fontsize = 16)
                    axs[i // 2, i % 2].grid(True)
                    axs[i // 2, i % 2].set_xlabel("K", fontsize = 12)
                    axs[i // 2, i % 2].set_xticks(np.linspace(1, K + 1, K // 2 + 1))
                    axs[i // 2, i % 2].set_yticks(np.linspace(0.55, 1, 10))
                    axs[i // 2, i % 2].legend(fontsize = 14)
                    axs[i // 2, i % 2].tick_params(labelsize = 12)
                fig.savefig(os.path.join(figDir, re.sub(" ", "", rankingColumn), "OptimalMetricScoresOverTopK_ " + targetFeatureType + "_Only_Features.png"))
            
            
            fig, axs = plt.subplots(2, 2, sharey = True, figsize = (24, 20), dpi = 100)
            for i in range(len(metrics)):
                metric = metrics[i]
                for j in range(len(models)):
                    model = models[j]
                    axs[i // 2, i % 2].plot([k for k in range(1, len(optimalScores[iRankingColumn][model][metric]) + 1)], optimalScores[iRankingColumn][model][metric], label = model)
                axs[i // 2, i % 2].set_title("Optimal " + metric + " Scores over Top-K " + targetFeatureType + "-Only Features", fontsize = 16)
                axs[i // 2, i % 2].grid(True)
                axs[i // 2, i % 2].set_xlabel("K", fontsize = 12)
                axs[i // 2, i % 2].legend(fontsize = 14)
                axs[i // 2, i % 2].tick_params(labelsize = 12)
            fig.savefig(os.path.join(figDir, re.sub(" ", "", rankingColumn), "OptimalMetricScoresOverTopK_ " + targetFeatureType + "_Only_FeaturesFull.png"))
        else:
            for K_Target in range(1, K + 1):
                for model in models:
                    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Untuned", "LOO", "TopKExperiments",  targetFeatureType, re.sub(" ", "", statsMethod), re.sub(" ", "", rankingColumn), model)
                    dfPredictions = pandas.read_csv(os.path.join(inputDir, model + "_" + targetFeatureType + "Top" + str(K_Target) + "_Predictions.csv"))
                    scores = {"Feature Combination" : [], "G : 0; P : 0" : [], "G : 0; P : 1" : [], "G : 1; P : 0" : [], "G : 1; P : 1" : [], "Accuracy" : [], "F1" : [], "Precision" : [], "Recall" : []}
                    for i in range(dfPredictions.shape[0]):
                        YPredict = dfPredictions.iloc[i, 1:].values.astype(int)
                        cm = confusion_matrix(YTrue, YPredict)
                        for j in range(cm.shape[0]):
                            for k in range(cm.shape[1]):
                                scores["G : " + str(j) + "; P : " + str(k)].append(cm[j, k])
                        scores["Feature Combination"].append(dfPredictions["Feature Combination"][i])
                        scores["Accuracy"].append(accuracy_score(YTrue, YPredict))
                        scores["F1"].append(f1_score(YTrue, YPredict))
                        scores["Precision"].append(precision_score(YTrue, YPredict))
                        scores["Recall"].append(recall_score(YTrue, YPredict))
                    # tuples = [(scores["Feature Combination"][i], scores["F1"][i]) for i in range(len(scores["Feature Combination"]))]
                    # tuples.sort(reverse = True, key = lambda x : x[1])
                    for metric in metrics:
                        optimalScores[iRankingColumn][model][metric].append(np.max(scores[metric]))
            fig, axs = plt.subplots(2, 2, sharey = True, figsize = (40, 20), dpi = 100)
            for i in range(len(metrics)):
                metric = metrics[i]
                for j in range(len(models)):
                    model = models[j]
                    axs[i // 2, i % 2].plot([k for k in range(1, K + 1)], optimalScores[iRankingColumn][model][metric], label = model)
                axs[i // 2, i % 2].set_title("Optimal " + metric + " Scores over Top-K " + targetFeatureType + " Features", fontsize = 16)
                axs[i // 2, i % 2].grid(True)
                axs[i // 2, i % 2].set_xlabel("K", fontsize = 12)
                axs[i // 2, i % 2].set_xticks(np.linspace(1, K + 1, K // 2 + 1))
                axs[i // 2, i % 2].set_yticks(np.linspace(0.55, 1, 10))
                axs[i // 2, i % 2].legend(fontsize = 14)
                axs[i // 2, i % 2].tick_params(labelsize = 12)
            fig.savefig(os.path.join(figDir, re.sub(" ", "", rankingColumn), "OptimalMetricScoresOverTopK_" + targetFeatureType + "_Features.png"))
    return optimalScores
    
def evaluateTopKExperiments():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["TX_0.25"]
    
    targetFeatureType = "RS"
    features = featuresRaw[FEATURE_TYPE_TO_FEATURES[targetFeatureType]]
    models = ["RF", "SVM"]
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    statsMethod = "Correlation"
    mode = "Classification"
    rankingColumns = ["Global", "LOO Median"]
    
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments", "Transformed", mode, "Untuned", targetFeatureType, re.sub(" ", "", statsMethod))
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    
    K = 100
    
    # optimalScores = evaluateTopKExperimentsHelper(K, YTrue, targetFeatureType, models, metrics, statsMethod, mode, rankingColumns, False, figDir)
    optimalScoresTargetOnly = evaluateTopKExperimentsHelper(K, YTrue, targetFeatureType, models, metrics, statsMethod, mode, rankingColumns[:1], True, figDir)
    
    # putting two together
    '''
    figTarget, axsTarget = plt.subplots(4, 2, figsize = (40, 64), dpi = 100)
    figTargetOnly, axsTargetOnly = plt.subplots(4, 2, figsize = (40, 64), dpi = 100)
    for iRankingColumn in range(len(rankingColumns)):
        rankingColumn = rankingColumns[iRankingColumn]
        for i in range(len(metrics)):
            metric = metrics[i]
            for j in range(len(models)):
                model = models[j]
                axsTarget[i, iRankingColumn].plot(optimalScores[iRankingColumn][model][metric][:K], label = model)
                axsTargetOnly[i, iRankingColumn].plot(optimalScoresTargetOnly[iRankingColumn][model][metric][:K], label = model)
            axsTarget[i, iRankingColumn].set_title("Optimal " + metric + " Scores over Top-K " + rankingColumn + " " + targetFeatureType + " Features", fontsize = 16)
            axsTargetOnly[i, iRankingColumn].set_title("Optimal " + metric + " Scores over Top-K " + rankingColumn + " " + targetFeatureType + "-Only Features", fontsize = 16)
            for axs in (axsTarget, axsTargetOnly):
                axs[i, iRankingColumn].grid(True)
                axs[i, iRankingColumn].set_xlabel("K", fontsize = 12)
                axs[i, iRankingColumn].set_xticks(np.linspace(0, K, K // 2 + 1))
                axs[i, iRankingColumn].set_yticks(np.linspace(0.55, 1, 10))
                axs[i, iRankingColumn].legend(fontsize = 14)
                axs[i, iRankingColumn].tick_params(labelsize = 12)
    figTarget.savefig(os.path.join(figDir, "OptimalMetricScoresOverTopK_ " + targetFeatureType + "_Features.png"))
    figTargetOnly.savefig(os.path.join(figDir, "OptimalMetricScoresOverTopK_ " + targetFeatureType + "_Only_Features.png"))
    '''

def ANOVA_F(X, Y):
    assert len(X) == len(Y)
    YUnique = np.unique(Y)
    N, K = len(Y), len(YUnique)
    data = {y : [] for y in YUnique}
    for i in range(len(X)):
        data[Y[i]].append(X[i, :])
    for y in data:
        data[y] = np.vstack(data[y])
    scores = np.zeros(X.shape[1], dtype = "float")
    means = {y : np.mean(data[y], axis = 0) for y in data}
    meanOverall = np.mean(X, axis = 0)
    
    numerator = 0.0 # explained variance
    denominator = 0.0 # between-group variability
    for y in data:
        numerator += len(data[y]) * np.power(means[y] - meanOverall, 2) / (K - 1)
        denominator += np.sum(np.power(np.subtract(data[y], means[y]), 2) / (N - K), axis = 0)
    return numerator / denominator

def testANOVAF():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    X = data["RS"][:, :2]
    YTrue = data["TX_0.25"]
    scores = ANOVA_F(X, YTrue)
    f, p = f_classif(X, YTrue)
    print(scores)
    print(f)



def evaluateFinalExperiments():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["TX_0.25"]
    
    
    mode = "Classification"
    targetFeatureType = "RS"
    statsMethod = "Correlation"
    rankingColumn = "Global"
    model = "RF"
    K = 21
    featureCombinations = []
    
    # RS included
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_v7", "Transformed", mode, "Untuned", "LOO", "TopKExperiments", targetFeatureType, re.sub(" ", "", statsMethod), re.sub(" ", "", rankingColumn), model)
    dfPredictions = pandas.read_csv(os.path.join(inputDir, model + "_Top" + str(K) + "RS_Predictions.csv"))
    scores = {"G : 0; P : 0" : [], "G : 0; P : 1" : [], "G : 1; P : 0" : [], "G : 1; P : 1" : [], "Accuracy" : [], "F1" : [], "Precision" : [], "Recall" : []}
    for i in range(dfPredictions.shape[0]):
        YPredict = dfPredictions.iloc[i, 1:].values.astype(int)
        cm = confusion_matrix(YTrue, YPredict)
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                scores["G : " + str(j) + "; P : " + str(k)].append(cm[j, k])
        scores["Accuracy"].append(accuracy_score(YTrue, YPredict))
        scores["F1"].append(f1_score(YTrue, YPredict))
        scores["Precision"].append(precision_score(YTrue, YPredict))
        scores["Recall"].append(recall_score(YTrue, YPredict))
    featureCombinations = dfPredictions["Feature Combination"].values.tolist()
    
    # RS excluded
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_v7", "Transformed", mode, "Untuned", "LOO", "TopKExperiments", targetFeatureType, re.sub(" ", "", statsMethod))
    dfPredictions = pandas.read_csv(os.path.join(inputDir, model + "_RSTop0_Predictions.csv"))
    for i in range(dfPredictions.shape[0]):
        YPredict = dfPredictions.iloc[i, 1:].values.astype(int)
        cm = confusion_matrix(YTrue, YPredict)
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                scores["G : " + str(j) + "; P : " + str(k)].append(cm[j, k])
        scores["Accuracy"].append(accuracy_score(YTrue, YPredict))
        scores["F1"].append(f1_score(YTrue, YPredict))
        scores["Precision"].append(precision_score(YTrue, YPredict))
        scores["Recall"].append(recall_score(YTrue, YPredict))
    featureCombinations += dfPredictions["Feature Combination"].values.tolist()
    
    M = {**{"Feature Combination" : featureCombinations}, **scores}
    dfOut = pandas.DataFrame(M)
    outputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_v7", "Transformed", mode, "Untuned", "LOO", "FinalExperiments")
    pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
    dfOut.to_csv(os.path.join(outputDir, model + "_RSTop" + str(K) + "_Scores.csv"), index = False)

def evaluateBoostingExperiments():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["TX_0.25"]
    
    model = "AdaBoost"
    mode = "Classification"
    targetFeatureType = "RS"
    statsMethod = "Correlation"
    rankingColumn = "Global"
    K = 16
    featureCombinations = []
    
    # top K RS + all predictions
    '''
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Untuned", "LOO", "BoostingExperiments")
    dfPredictions = pandas.read_csv(os.path.join(inputDir, model + "_Top" + str(K) + "RS_Predictions.csv"))
    scores = {"G : 0; P : 0" : [], "G : 0; P : 1" : [], "G : 1; P : 0" : [], "G : 1; P : 1" : [], "Accuracy" : [], "F1" : [], "Precision" : [], "Recall" : []}
    for i in range(dfPredictions.shape[0]):
        YPredict = dfPredictions.iloc[i, 1:].values.astype(int)
        cm = confusion_matrix(YTrue, YPredict)
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                scores["G : " + str(j) + "; P : " + str(k)].append(cm[j, k])
        scores["Accuracy"].append(accuracy_score(YTrue, YPredict))
        scores["F1"].append(f1_score(YTrue, YPredict))
        scores["Precision"].append(precision_score(YTrue, YPredict))
        scores["Recall"].append(recall_score(YTrue, YPredict))
    featureCombinations = dfPredictions["Feature Combination"].values.tolist()
    optimalIndex = np.argmax(scores["F1"])
    print(str(featureCombinations[optimalIndex]) + "  " + str(scores["F1"][optimalIndex]))
    '''
    
    # no RS, finetuned
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Finetuned", "LOO", "AdaBoost")
    dfPredictions = pandas.read_csv(os.path.join(inputDir, model + "_NoRS_Predictions.csv"))
    scores = {"G : 0; P : 0" : [], "G : 0; P : 1" : [], "G : 1; P : 0" : [], "G : 1; P : 1" : [], "Accuracy" : [], "F1" : [], "Precision" : [], "Recall" : []}
    for i in range(dfPredictions.shape[0]):
        YPredict = dfPredictions.iloc[i, 1:].values.astype(int)
        cm = confusion_matrix(YTrue, YPredict)
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                scores["G : " + str(j) + "; P : " + str(k)].append(cm[j, k])
        scores["Accuracy"].append(accuracy_score(YTrue, YPredict))
        scores["F1"].append(f1_score(YTrue, YPredict))
        scores["Precision"].append(precision_score(YTrue, YPredict))
        scores["Recall"].append(recall_score(YTrue, YPredict))
    featureCombinations = dfPredictions["Feature Combination"].values.tolist()
    M = {**{"Feature Combination" : featureCombinations}, **scores}
    dfOut = pandas.DataFrame(M)
    outputDir = inputDir
    dfOut.to_csv(os.path.join(outputDir, model + "_NoRS_Scores.csv"), index = False)
    
    # param grid
    '''
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments", "Transformed", "Classification", "Untuned")
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    inputDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments", "Transformed", mode, "Untuned", "LOO", "AdaBoostParameterSearch")
    df = pandas.read_csv(os.path.join(inputDir, "AdaBoost_Grid_Results_2.csv"))
    grid = {"n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "learning_rate" : [1e-3, 1e-2, 1e-1, 1]}
    grid = {"n_estimators" : [100 + 50 * i for i in range(19)],
            "learning_rate" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    params = [*grid]
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
    F1Scores = np.round(F1Scores, 4)
    fig, ax = makeMatrixPlot(M = F1Scores.T, title = "F1 Scores over Parameter Grid",\
                            xTickLabels = grid["n_estimators"], yTickLabels = grid["learning_rate"],\
                            xLabel = "Iterations", yLabel = "Learning Rate",\
                            threshold = "auto", cmap = plt.cm.Blues, vmin = 0, vmax = 1,\
                            figsize = (16, 8), dpi = 100,\
                            integer = False)
    fig.savefig(os.path.join(figDir, "F1OverAdaBoostParamGrid2.png"))
    '''

def makeAQStats():
    data, featuresRaw = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["AQ"].squeeze()
    
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments", "Transformed", "Statistics", "AQ")
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    
    # AQ histogram
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.hist(YTrue, bins = np.linspace(0, 1, 11), edgecolor = "k")
    ax.set_xlabel("AQ", fontsize = 14)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_title("AQ Histogram", fontsize = 16)
    ax.text(0.16, 12, "Mean = " + str(np.round(np.mean(YTrue), 2)), fontsize = 12)
    ax.text(0.16, 11.6, "Median = " + str(np.round(np.median(YTrue), 2)), fontsize = 12)
    ax.text(0.16, 11.2, "Std. = " + str(np.round(np.std(YTrue), 2)), fontsize = 12)
    fig.savefig(os.path.join(figDir, "AQHistogram.png"))
    
    
    featureTypes = ["CS", "DM", "LS", "PS_W", "FA", "PS_G", "RS"]
    # continuous
    fig, axs = plt.subplots(len(featureTypes), 1, figsize = (12, 28))
    for i in range(len(featureTypes)):
        correlations = []
        for j in range(data[featureTypes[i]].shape[1]):
            X = data[featureTypes[i]][:, j]
            if len(np.unique(X)) > 1:
                r, p = sp.stats.pearsonr(data[featureTypes[i]][:, j], YTrue)
                correlations.append(p)
        axs[i].hist(correlations, bins = np.linspace(0, 1, 11), edgecolor = "k")
        axs[i].set_xlabel("Correlation", fontsize = 14)
        axs[i].set_xticks(np.linspace(0, 1, 11))
        axs[i].set_title(featureTypes[i] + " - Continuous AQ Correlation Histogram", fontsize = 16)
        axs[i].grid(True)
        if featureTypes[i] == "DM":
            print(correlations)
    fig.tight_layout()
    fig.savefig(os.path.join(figDir, "AQCorrelationHistograms.png"))
    
    
    # binary
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        Y = (YTrue > threshold).astype("int")
        counts = {}
        for y in Y:
            counts[y] = 1 + counts.get(y, 0)
        print("Threshold = " + str(threshold) + "  Label distribution : " + str(counts))
        fig, axs = plt.subplots(len(featureTypes), 1, figsize = (12, 28))
        for i in range(len(featureTypes)):
            correlations = []
            for j in range(data[featureTypes[i]].shape[1]):
                X = data[featureTypes[i]][:, j]
                if len(np.unique(X)) > 1:
                    r, p = sp.stats.pearsonr(data[featureTypes[i]][:, j], Y)
                    correlations.append(p)
            axs[i].hist(correlations, bins = np.linspace(0, 1, 11), edgecolor = "k")
            axs[i].set_xlabel("Correlation", fontsize = 14)
            axs[i].set_xticks(np.linspace(0, 1, 11))
            axs[i].set_title(featureTypes[i] + " - Binary AQ (T = " + str(threshold) + ") Correlation Histogram", fontsize = 16)
            axs[i].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(figDir, "AQ(T=" + str(threshold) + ")CorrelationHistograms.png"))
    
def compareV6V7CorrelationGlobalRankings():
    versions = ("v6", "v7")
    dfs = {v : pandas.read_excel(os.path.join(CROSS_SITE_DATASET, "compiled_dataset_RSbivariate_without_controls_" + v + ".xlsx"), header = [0, 1]) for v in versions}
    matrixRS = {v : dfs[v]["restingstate_bivariate_correlations"].values for v in versions}
    for v in versions:
        matrixRS[v] = (matrixRS[v] - matrixRS[v].min(axis = 0)) / (matrixRS[v].max(axis = 0) - matrixRS[v].min(axis = 0))
    featuresRS = {v : [col for col in dfs[v]["restingstate_bivariate_correlations"]] for v in versions}
    Y = dfs["v6"][('behavioral', "tx_change_categorical_0.25_bd")].values
    tuples = {v : [] for v in versions}
    for v in versions:
        X = matrixRS[v]
        for j in range(X.shape[1]):
            r, _ = sp.stats.pearsonr(X[:, j].squeeze(), Y)
            r = np.round(r, 4)
            tuples[v].append((featuresRS[v][j], r))
        tuples[v].sort(reverse = True, key = lambda x : abs(x[1]))
    tuples["v6"] += [None] * (len(tuples["v7"]) - len(tuples["v6"]))
    dfOut = pandas.DataFrame({"Global " + v : tuples[v] for v in versions})
    dfOut.to_csv("CorrelationRankingsV6V7.csv", index = False)

def evaluateFeatureRankingExperiments():
    data, features = getData(os.path.join(CROSS_SITE_DATASET, DATA_FILENAME), normalize = True)
    YTrue = data["TX_0.25"]
    model = "RF"
    tableDir = os.path.join(PATH_TO_TABLES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "LOO", "FeatureRankingExperiments", model)
    K_ReliefF = [1, 50, 100, 150, 200, 250, 300, 350, 400, "SURF"]
    f1Scores = {}
    K = 51
    for k_ReliefF in K_ReliefF:
        if k_ReliefF != "SURF":
            dfPredictions = pandas.read_csv(os.path.join(tableDir, model + "_ReliefF(" + str(k_ReliefF) + ")_Predictions.csv"))
            f1Scores["ReliefF(" + str(k_ReliefF) + ")"] = []
            for i in range(K):
                YPredict = dfPredictions.iloc[i, 1:].values.astype("int")
                f1Scores["ReliefF(" + str(k_ReliefF) + ")"].append(f1_score(YTrue, YPredict))
        else:
            dfPredictions = pandas.read_csv(os.path.join(tableDir, model + "_SURF_Predictions.csv"))
            f1Scores["SURF"] = []
            for i in range(K):
                YPredict = dfPredictions.iloc[i, 1:].values.astype("int")
                f1Scores["SURF"].append(f1_score(YTrue, YPredict))
    fig, ax = plt.subplots(figsize = (16, 12))
    for key in f1Scores:
        ax.plot(f1Scores[key], label = key)
    ax.set_title("ReliefF Methods " + model + " F1 Scores over Top K Connectivity Scores", fontsize = 16)
    ax.set_xlabel("K")
    ax.set_xticks(np.linspace(0, K - 1, K).astype("int"))
    ax.set_xticklabels(np.linspace(0, K - 1, K).astype("int"))
    ax.grid(True)
    ax.legend()
    figDir = os.path.join(PATH_TO_FIGURES, "AllFeaturesExperiments_" + DATA_VERSION, "Transformed", "Classification", "Untuned", "FeatureRankingExperiments")
    pathlib.Path(figDir).mkdir(parents = True, exist_ok = True)
    fig.savefig(os.path.join(figDir, "ReliefFMethods_" + model + "_F1Scores.png"))

if __name__ == "__main__":
    # RSStatisticsAnalysis()
    # evaluateTopKExperiments()
    # testANOVAF()
    # evaluateSVMParamSearchExperiments()
    # evaluateRFParamSearchExperiments()
    # evaluateAdaBoostParamSearchExperiments()
    evaluateSVMRFAdaBoost()
    
    # evaluateFinalExperiments()
    # evaluateBoostingExperiments()
    # makeAQStats()
    # evaluateFeatureRankingExperiments()