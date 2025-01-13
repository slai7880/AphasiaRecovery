from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, r2_score, mean_squared_error
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing
from itertools import combinations
import sklearn.feature_selection as fs
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, SpectralEmbedding
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import bisect
from sklearn.feature_selection import chi2, f_classif

from sklearn.cluster import KMeans, DBSCAN, OPTICS

import matplotlib.pyplot as plt

from common import *

N_PROCESSES = 2





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

def recoverCorrelationMatrix(features, values, regions, regionIndices):
    M = np.matrix([[0.0] * len(regions) for _ in range(len(regions))])
    for i in range(len(features)):
        featureSplit = features[i].split(" vs ")
        region1 = featureSplit[0]
        region2 = featureSplit[1][:len(featureSplit[1]) - 3]
        M[regionIndices[region1], regionIndices[region2]] = values[i]
    return M

def makeTargetCorrelationMatrices(targetCorrelationTuples, matrixTypes, targets, regions, regionIndices, threshold = 0):
    matrices = {t : {m : None for m in matrixTypes} for t in targets}
    for i in range(len(targets)):
        for j in range(len(matrixTypes)):
            matrixType, target = matrixTypes[j], targets[i]
            tuples = targetCorrelationTuples[target][matrixType]
            features = [tuples[k][0] for k in range(len(tuples))]
            values = [tuples[k][1] for k in range(len(tuples))]
            M = recoverCorrelationMatrix(features, values, regions, regionIndices)
            if matrixType == "Bivariate":
                M += M.T
            matrices[target][matrixType] = M
    for j in range(len(matrixTypes)):
        matrixType = matrixTypes[j]
        M1 = matrices[targets[0]][matrixType]
        M2 = matrices[targets[1]][matrixType]
        mask = np.multiply(np.abs(M1) >= threshold, np.abs(M2) >= threshold)
        matrices[targets[0]][matrixType] = np.multiply(M1, mask)
        matrices[targets[1]][matrixType] = np.multiply(M2, mask)
    return matrices

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


def makeCorrelationPlots(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX, threshold = 0):
    # get the original matrix axis
    # dfExample = pandas.read_excel(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "BU01.xlsx"))
    # regions = dfExample.columns[1:].tolist()
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    

    
    targetCorrelationTuples = makeTargetCorrelationTuples(matrixTypes, targets, labelColumnIndex)
    
    targetCorrelationMatrices = makeTargetCorrelationMatrices(targetCorrelationTuples, matrixTypes, targets, regions, regionIndices, threshold)
    targetCorrelationHistograms = makeTargetCorrelationHistograms(targetCorrelationTuples, matrixTypes, targets, threshold)
    
    
    
    for matrixType in matrixTypes:
        correlations = {t : [] for t in targets}
        minCorr = []
        regions = {"Region 1" : [], "Region 2" : []}
        tuplesAQ = targetCorrelationTuples["AQ"][matrixType]
        tuplesTX = targetCorrelationTuples["TX"][matrixType]
        for i in range(len(tuplesAQ)):
            correlations["AQ"].append(tuplesAQ[i][1])
            correlations["TX"].append(tuplesTX[i][1])
            minCorr.append(min(abs(tuplesAQ[i][1]), abs(tuplesTX[i][1])))
            featureSplit = tuplesAQ[i][0].split(" vs ")
            region1 = featureSplit[0]
            region2 = featureSplit[1][:len(featureSplit[1]) - 3]
            regions["Region 1"].append(region1)
            regions["Region 2"].append(region2)
        dfOut = pandas.DataFrame({"Region 1" : regions["Region 1"], "Region 2" : regions["Region 2"],
                                    "Correlation AQ" : correlations["AQ"], "Correlation TX" : correlations["TX"],
                                    "Minimum Absolute Correlation" : minCorr})
        outputDir = os.path.join(TABLE_DIR, "Correlation")
        pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
        dfOut.to_csv(os.path.join(outputDir, "CorrelationTable(" + matrixType + ").csv"), index = None)
    
            
    
    cmap = COLOR_MAP
    
    for j in range(len(matrixTypes)):
        matrixType = matrixTypes[j]
        # figHeatmap, axsHeatmap = plt.subplots(1, 2, figsize = (40, 20), dpi = 100)
        if matrixType == "Bivariate":
            figBar, axsBar = plt.subplots(1, 2, figsize = (40, 20), dpi = 100)
            figBarh, axsBarh = plt.subplots(1, 2, figsize = (60, 80), dpi = 100)
        else:
            figBar, axsBar = plt.subplots(1, 2, figsize = (40, 20), dpi = 100)
            figBarh, axsBarh = plt.subplots(1, 2, figsize = (60, 140), dpi = 100)
        
        outputDir = os.path.join(FIGURE_DIR, "Correlation", "Thresholding", str(threshold), matrixType)
        pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
        for i in range(len(targets)):
            target = targets[i]
            
            # heatmaps
            '''
            matrix = targetCorrelationMatrices[target][matrixType]
            axsHeatmap[i].imshow(matrix, origin = "upper", cmap = cmap, vmin = -0.5, vmax = 0.5)
            axsHeatmap[i].set_title(matrixType + "-" + target + " Correlation Heatmap", fontsize = 36)
            axsHeatmap[i].set_xticks([i for i in range(len(regions))])
            axsHeatmap[i].set_yticks([i for i in range(len(regions))])
            axsHeatmap[i].set_xticklabels(regions, fontsize = 14)
            axsHeatmap[i].set_yticklabels(regions, fontsize = 14)
            axsHeatmap[i].xaxis.tick_top()
            plt.setp(axsHeatmap[i].get_xticklabels(), rotation = 90, ha = "left", rotation_mode = "anchor")
            '''
            
            # barh plots
            histogram = targetCorrelationHistograms[target][matrixType]
            features, values, colors = histogram["features"], histogram["values"], histogram["colors"]
            axsBar[i].bar(features, values, color = colors)
            axsBar[i].set_title(matrixType + "-" + target + " Correlation Scores (Sorted by Magnitude)", fontsize = 36)
            axsBar[i].set_ylim(0, 0.5)
            axsBar[i].set_xticklabels([])
            
            
            # barh plots
            histogram = targetCorrelationHistograms[target][matrixType]
            features, values, colors = histogram["features"], histogram["values"], histogram["colors"]
            axsBarh[i].barh(features, values, color = colors)
            axsBarh[i].invert_yaxis()
            axsBarh[i].set_title(matrixType + "-" + target + " Correlation Scores (Sorted by Magnitude)", fontsize = 36)
            axsBarh[i].set_xlim(0, 0.5)
            axsBarh[i].set_yticklabels(features, fontsize = 8)
            
            
        # figHeatmap.tight_layout()
        '''
        figHeatmap.subplots_adjust(right = 0.9)
        cbar_ax = figHeatmap.add_axes([0.92, 0.14, 0.02, 0.72])
        sm = plt.cm.ScalarMappable(norm = mcolors.Normalize(vmin = -0.5, vmax = 0.5), cmap = cmap)
        figHeatmap.colorbar(sm, cax = cbar_ax)
        figHeatmap.savefig(os.path.join(outputDir, "CorrelationHeatmaps.png"))
        plt.close(figHeatmap)
        '''
        
        figBar.tight_layout()
        figBar.savefig(os.path.join(outputDir, "CorrelationBarplots.png"))
        plt.close(figBar)
        
        figBarh.tight_layout()
        figBarh.savefig(os.path.join(outputDir, "CorrelationBarhplots.png"))
        plt.close(figBarh)
        
        # scatter plot
        '''
        AQs = []
        TXs = []
        for i in range(len(targetCorrelationTuples[targets[0]][matrixType])):
            tupleAQ = targetCorrelationTuples[targets[0]][matrixType][i]
            tupleTX = targetCorrelationTuples[targets[1]][matrixType][i]
            if tupleAQ[3] >= threshold and tupleTX[3] >= threshold:
                AQs.append(tupleAQ[1])
                TXs.append(tupleTX[1])
        figScatter, axScatter = plt.subplots(figsize = (10, 10), dpi = 100)
        axScatter.scatter(AQs, TXs, color = "g")
        # axScatter.set_title("Scatter Plot of AQ vs TX Scores", fontsize = 18)
        axScatter.set_xlabel("AQ", horizontalalignment = 'right', x = 1.0, fontsize = 16)
        axScatter.set_ylabel("TX", verticalalignment = 'top', y = 1.0, rotation = 0, fontsize = 16)
        axScatter.set_xlim(-0.5, 0.5)
        axScatter.set_ylim(-0.5, 0.5)
        
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        axScatter.spines['left'].set_position('zero')
        axScatter.spines['bottom'].set_position('zero')
        
        # Eliminate upper and right axes
        axScatter.spines['right'].set_color('none')
        axScatter.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        axScatter.xaxis.set_ticks_position('bottom')
        axScatter.yaxis.set_ticks_position('left')
        figScatter.tight_layout()
        figScatter.savefig(os.path.join(outputDir, "CorrelationScatter.png"))
        plt.close(figScatter)
        '''
    

def makeManifoldPlots(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX):
    manifolds = {"tSNE" : TSNE,
                 "LLE" : LocallyLinearEmbedding,
                 "Isomap" : Isomap,
                 "SpectralEmbedding" : SpectralEmbedding}
    cmap = COLOR_MAP
    for name in manifolds:
        fig, axs = plt.subplots(2, 2, figsize = (20, 20), dpi = 100)
        for i in range(len(targets)):
            for j in range(len(matrixTypes)):
                matrixType, target = matrixTypes[j], targets[i]
                filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
                filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
                _, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
                method = manifolds[name](n_components = 2)
                XReduced = method.fit_transform(X)
                tuples = []
                for k in range(len(Y)):
                    tuples.append((Y[k], XReduced[k, 0], XReduced[k, 1]))
                tuples.sort(key = lambda x : x[0])
                X = [t[1] for t in tuples]
                Y = [t[2] for t in tuples]
                t = np.arange(len(Y))
                axs[i, j].scatter(X, Y, c = t, cmap = cmap)
                axs[i, j].set_title(name + " Scatter Plot (" + matrixType + ", " + target + ")", fontsize = 18)
        fig.subplots_adjust(right = 0.9)
        cbar_ax = fig.add_axes([0.92, 0.14, 0.02, 0.72])
        sm = plt.cm.ScalarMappable(norm = mcolors.Normalize(vmin = 0, vmax = 1), cmap = cmap)
        fig.colorbar(sm, cax = cbar_ax)
        outputDir = os.path.join(FIGURE_DIR, "Correlation", "Manifold")
        pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
        fig.savefig(os.path.join(outputDir, name + "Plots.png"))

def makeIndividualRegionConnectivityPlots(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX):
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    XBounds = {"Bivariate" : (-1, 2), "Semipartial" : (-0.4, 0.7)}
    if REVERSE_TRANSFORM:
        XBounds = {"Bivariate" : (-1, 1), "Semipartial" : (-1, 1)}
    cmap = COLOR_MAP
    outputDir = os.path.join(FIGURE_DIR, "Connectivity", "IndividualHeatmaps")
    for matrixType in matrixTypes:
        filename = "RS_" + matrixType.lower() + "_" + targets[0] + "_continuous.xlsx"
        filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
        IDs, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
        for i in range(len(IDs)):
            matrix = recoverCorrelationMatrix(XHeaders, X[i, :], regions, regionIndices)
            if matrixType == "Bivariate":
                matrix += matrix.T
            fig, ax = plt.subplots(figsize = (10, 10), dpi = 100)
            im = ax.imshow(matrix, origin = "upper", cmap = cmap, vmin = XBounds[matrixType][0], vmax = XBounds[matrixType][1])
            ax.set_title(matrixType + " Connectivity Heatmap", fontsize = 16)
            ax.set_xticks([i for i in range(len(regions))])
            ax.set_yticks([i for i in range(len(regions))])
            ax.set_xticklabels(regions, fontsize = 8)
            ax.set_yticklabels(regions, fontsize = 8)
            ax.xaxis.tick_top()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size = "5%", pad = 0.2)
            fig.colorbar(im, cax = cax)
            plt.setp(ax.get_xticklabels(), rotation = 90, ha = "left", rotation_mode = "anchor")
            pathlib.Path(os.path.join(outputDir, IDs[i])).mkdir(parents = True, exist_ok = True)
            fig.savefig(os.path.join(outputDir, IDs[i], matrixType + ".png"))
            plt.close(fig)

def makeMeanRegionConnectivityPlots(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX):
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    averageConnectivities = {m : None for m in matrixTypes}
    averageAbsConnectivities = {m : None for m in matrixTypes}
    
    for matrixType in matrixTypes:
        filename = "RS_" + matrixType.lower() + "_" + targets[0] + "_continuous.xlsx"
        filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
        _, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
        features, values, valuesAbs = XHeaders, np.mean(X, axis = 0), np.mean(np.abs(X), axis = 0)
        print(str(X.min()) + "  " + str(X.max()))
        print()
        M = recoverCorrelationMatrix(features, values, regions, regionIndices)
        MAbs = recoverCorrelationMatrix(features, valuesAbs, regions, regionIndices)
        if matrixType == "Bivariate":
            M += M.T
            MAbs += MAbs.T
        averageConnectivities[matrixType] = M
        averageAbsConnectivities[matrixType] = MAbs
        df = pandas.DataFrame(M, columns = regions)
        df[""] = regions
        df = df[[""] + regions]
        df.to_csv(os.path.join(TABLE_DIR, "Connectivity", "AverageConnectivityMatrix.csv"), index = None)
        
        df = pandas.DataFrame(MAbs, columns = regions)
        df[""] = regions
        df = df[[""] + regions]
        df.to_csv(os.path.join(TABLE_DIR, "Connectivity", "AverageAbsoluteConnectivityMatrix.csv"), index = None)
    
    outputDir = os.path.join(FIGURE_DIR, "Connectivity")
    cmap = COLOR_MAP
    bounds = {"Bivariate" : (-0.3, 1.1), "Semipartial" : (-0.1, 0.4)}
    if REVERSE_TRANSFORM:
        bounds = {"Bivariate" : (-1, 1), "Semipartial" : (-1, 1)}
    for i in range(len(matrixTypes)):
        matrixType = matrixTypes[i]
        figC, axsC = plt.subplots(1, 2, figsize = (40, 20), dpi = 100)
        axsC[0].imshow(averageConnectivities[matrixType], origin = "upper", cmap = cmap, vmin = bounds[matrixType][0], vmax = bounds[matrixType][1])
        axsC[0].set_title(matrixType + " Connectivity Heatmap", fontsize = 36)
        axsC[0].set_xticks([i for i in range(len(regions))])
        axsC[0].set_yticks([i for i in range(len(regions))])
        axsC[0].set_xticklabels(regions, fontsize = 16)
        axsC[0].set_yticklabels(regions, fontsize = 16)
        axsC[0].xaxis.tick_top()
        plt.setp(axsC[0].get_xticklabels(), rotation = 90, ha = "left", rotation_mode = "anchor")
        
        axsC[1].imshow(averageAbsConnectivities[matrixType], origin = "upper", cmap = cmap, vmin = bounds[matrixType][0], vmax = bounds[matrixType][1])
        axsC[1].set_title(matrixType + " Absolute Connectivity Heatmap", fontsize = 36)
        axsC[1].set_xticks([i for i in range(len(regions))])
        axsC[1].set_yticks([i for i in range(len(regions))])
        axsC[1].set_xticklabels(regions, fontsize = 16)
        axsC[1].set_yticklabels(regions, fontsize = 16)
        axsC[1].xaxis.tick_top()
        plt.setp(axsC[1].get_xticklabels(), rotation = 90, ha = "left", rotation_mode = "anchor")
        
        figC.subplots_adjust(right = 0.9)
        cbar_ax = figC.add_axes([0.92, 0.14, 0.02, 0.72])
        sm = plt.cm.ScalarMappable(norm = mcolors.Normalize(vmin = bounds[matrixType][0], vmax = bounds[matrixType][1]), cmap = cmap)
        figC.colorbar(sm, cax = cbar_ax)
        figC.savefig(os.path.join(outputDir, matrixType + "ConnectivityHeatmaps.png"))
    

def makeExampleFeatureTargetPlots(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX):
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    targetCorrelationTuples = makeTargetCorrelationTuples(matrixTypes, targets, labelColumnIndex)
    targetCorrelationHistograms = makeTargetCorrelationHistograms(targetCorrelationTuples, matrixTypes, targets, threshold = 0)
    
    for i in range(len(targets)):
        for j in range(len(matrixTypes)):
            target, matrixType = targets[i], matrixTypes[j]
            outputDir = os.path.join(FIGURE_DIR, "Correlation", "FeatureTargetCorrelationScatterPlots", matrixType, target)
            pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
            filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
            filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
            _, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
            XHeaders = XHeaders.tolist()
            for k in range(len(XHeaders)):
                r, p = sp.stats.pearsonr(np.array(X[:, k].reshape(1, -1).squeeze()), Y)
                color = "m"
                if r < 0:
                    color = "c"
                fig, ax = plt.subplots(figsize = (8, 8), dpi = 100)
                ax.scatter(np.array(X[:, k].reshape(1, -1).squeeze()), Y, color = color, label = str(np.round(r, 2)))
                ax.set_title(matrixType + " (" + XHeaders[k] + ") - " + target + " Correlation")
                ax.set_xlabel(XHeaders[k], fontsize = 12)
                ax.set_ylabel(target, fontsize = 12)
                ax.legend()
                fig.savefig(os.path.join(outputDir, XHeaders[k] + ".png"))
                plt.close(fig)

def getJointTopRankingCounts(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX, reverseTransform = REVERSE_TRANSFORM):
    regions = None
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    tuples = {}
    for matrixType in matrixTypes:
        filename = "RS_" + matrixType.lower() + "_AQ_continuous.xlsx"
        filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
        _, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
        if reverseTransform:
            X = np.tanh(X)
        XAbs = np.abs(X)
        XAbsMean = np.mean(XAbs, axis = 0).squeeze()
        XAbsMeanMatrix = recoverCorrelationMatrix(XHeaders, XAbsMean, regions, regionIndices)
        tuples[matrixType] = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                value = max(XAbsMeanMatrix[i, j], XAbsMeanMatrix[j, i])
                if value > 0:
                    tuples[matrixType].append((regions[i], regions[j], value))
        tuples[matrixType].sort(reverse = True, key = lambda x : x[2])
    # print(len(tuples[matrixTypes[0]]) == len(tuples[matrixTypes[1]]))
    counts = []
    for i in range(1, len(tuples[matrixTypes[0]]) + 1):
        regionPairs = {}
        for matrixType in matrixTypes:
            regionPairs[matrixType] = [t[0] + " " + t[1] for t in tuples[matrixType][:i]]
        intersection = set(regionPairs[matrixTypes[0]]) & set(regionPairs[matrixTypes[1]])
        counts.append(len(intersection))
    return counts

def makeJointTopRankingPlots(matrixTypes = MATRIX_TYPES, targets = TARGETS, labelColumnIndex = LABEL_COLUMN_INDEX):
    counts = getJointTopRankingCounts(matrixTypes, targets, labelColumnIndex, reverseTransform = False)
    fig, ax = plt.subplots(figsize = (10, 8), dpi = 100)
    ax.plot([i for i in range(len(counts))], counts)
    ax.set_title("Sizes of Intersection Between Sorted Bivariate and Semipartial Connectivity")
    ax.set_xlabel("Number of Region Pairs")
    fig.savefig(os.path.join(PATH_TO_FIGURES, "RSExperiments", "Transformed", "Connectivity", "IntersectionSizePlot.png"))
    fig.savefig(os.path.join(PATH_TO_FIGURES, "RSExperiments", "ReverseTransformed", "Connectivity", "IntersectionSizePlot.png"))

def makeUnivariateStatPlots(matrixType, target):
    regions = None
    file = open(os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", "AllRegions"), "r")
    for line in file:
        regions = eval(line)
    file.close()
    regionIndices = {regions[i] : i for i in range(len(regions))}
    tuples = {}
    filename = "RS_" + matrixType.lower() + "_AQ_continuous.xlsx"
    filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
    _, X, XHeaders, Y = getCorrelationData(filepath, LABEL_COLUMN_INDEX)
    print(X.shape)

def runConnectivityAnalysis():
    # makeIndividualRegionConnectivityPlots(matrixTypes, targets, labelColumnIndex)
    # makeMeanRegionConnectivityPlots(matrixTypes, targets, labelColumnIndex)
    # makeManifoldPlots(matrixTypes, targets, labelColumnIndex)
    makeJointTopRankingPlots(matrixTypes, targets, labelColumnIndex)

def runCorrelationAnalysis():
    thresholds = [np.round(i * 0.05, 2) for i in range(10)]
    
    # makeCorrelationPlots(matrixTypes, targets, labelColumnIndex, 0.3)
    
    
    for t in thresholds:
        print(t)
        makeCorrelationPlots(matrixTypes, targets, labelColumnIndex, t)
    
    
    
    makeExampleFeatureTargetPlots(matrixTypes, targets, labelColumnIndex)

def runClusteringAnalysis():
    methodNames = ["DBSCAN", "OPTICS"]
    methods = {"DBSCAN" : DBSCAN, "OPTICS" : OPTICS}
    outputDir = os.path.join(PATH_TO_FIGURES, "RSExperiments", "Connectivity", "Clustering")
    pathlib.Path(outputDir).mkdir(parents = True, exist_ok = True)
    for i in range(len(targets)):
        for j in range(len(matrixTypes)):
            target, matrixType = targets[i], matrixTypes[j]
            filename = "RS_" + matrixType.lower() + "_" + target + "_continuous.xlsx"
            filepath = os.path.join(CROSS_SITE_DATASET, "resting_state_data", "for_simple_correlations", filename)
            _, X, XHeaders, Y = getCorrelationData(filepath, labelColumnIndex)
            if target == "TX":
                Y *= 100
            for k in range(len(methodNames)):
                method = methods[methodNames[k]](metric = "correlation")
                method.fit(X)
                allLabels = list(set(method.labels_))
                NLabels = len(allLabels)
                print(matrixType + "  " + target + "  " + methodNames[k])
                print("Number of unique labels = " + str(NLabels))
                print()
                if NLabels > 1:
                    histograms = {L : [0] * 11 for L in allLabels}
                    multiplier = 1
                    xTickLabels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                    for l in range(len(Y)):
                        histograms[method.labels_[l]][int((Y[l] * multiplier) // 10)] += 1
                    fig, axs = plt.subplots(NLabels, 1, figsize = (12, 10), dpi = 100)
                    for l in range(len(allLabels)):
                        label = allLabels[l]
                        axs[l].bar(xTickLabels, histograms[label])
                        axs[l].set_title(methodNames[k] + " Clustering Histogram (" + matrixType + ", " + target + ")")
                        xlabel = target
                        if target == "TX":
                             xlabel += " (scaled)"
                        axs[l].set_xlabel(xlabel)
                        axs[l].set_xticks(xTickLabels)
                        axs[l].set_xticklabels(xTickLabels)
                    fig.tight_layout()
                    fig.savefig(os.path.join(outputDir, methodNames[k] + "ClusteringHistogram" + matrixType + target + ".png"))
                    plt.close(fig)

if __name__ == "__main__":
    # runCorrelationAnalysis()
    # runClusteringAnalysis()
    # runConnectivityAnalysis()
    '''
    tuples = makeTargetCorrelationTuples()
    optimalThresholds = {"AQ" : {"Bivariate" : 0.33, "Semipartial" : 0.33}, "TX" : {"Bivariate" : 0.31, "Semipartial" : 0.35}}
    features = {t : {m : [] for m in MATRIX_TYPES} for t in TARGETS}
    for target in TARGETS:
        L = 0
        for matrixType in MATRIX_TYPES:
            for i in range(len(tuples[target][matrixType])):
                if tuples[target][matrixType][i][3] >= optimalThresholds[target][matrixType]:
                    features[target][matrixType].append(tuples[target][matrixType][i][0])
            print(target + "  " + matrixType + " (" + str(optimalThresholds[target][matrixType]) + ") : " + str(features[target][matrixType]))
            L += len(features[target][matrixType])
        print(L)
    '''
    matrixType, target = "Bivariate", "TX"
    makeUnivariateStatPlots(matrixType, target)
    