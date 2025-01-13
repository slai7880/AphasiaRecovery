import sys, os, pathlib
import numpy as np
import pandas
import matplotlib.pyplot as plt
import argparse

FEATURE_COLOR = {"LS" : "blue", "PD" : "green", "FA" : "red", "DM" : "gray", "SP" : "orange"}
LETTER_MAP = {c : ord(c) - ord("A") + 1 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

def initializeParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default = None, type = str, required = True)
    parser.add_argument("--FSScore_path", default = None, type = str, required = True)
    parser.add_argument("--feature_combination", default = None, type = str, required = True)
    parser.add_argument("--column", default = None, type = str, required = True)
    parser.add_argument("--fig_path", default = None, type = str)
    parser.add_argument("--sort_scores", action = "store_true")
    return parser

def getScores(df, features, header):
    if pandas.isnull(df[header][df.iloc[:, 0].values.tolist().index(features)]):
        print("NaN")
        sys.exit()
    else:
        return eval(df[header][df.iloc[:, 0].values.tolist().index(features)])

def makeFSScorePlot(dataPath, FSScorePath, features, column, figPath, sortScores):
    features = features.split("+")
    for i in range(len(features)):
        features[i] = features[i].strip()
    columnIndex = 0
    for c in column:
        columnIndex = columnIndex * len(LETTER_MAP) + LETTER_MAP[c]
    columnIndex -= 1
    
    
    df = pandas.read_excel(dataPath, header = [0, 1])
    headers = [col for col in df]
    headerGroupMap = {}
    headerColorMap = {}
    for h in headers[2:]:
        if h[0] != 'Behavioral Data':
            if h[0] == "Lesion size":
                abbr = "LS"
            elif h[0] == "Proportion Disconnection":
                abbr = "PD"
            elif h[0] == "Average FA values":
                abbr = "FA"
            elif h[0] == "Demographic info":
                abbr = "DM"
            elif h[0] == "Percent spared per region":
                abbr = "SP"
            elif h[0] == "Resting state":
                abbr = "RS"
            if not abbr in headerGroupMap:
                headerGroupMap[abbr] = []
                headerColorMap[abbr] = []
            headerGroupMap[abbr].append(h[1])
            headerColorMap[abbr].append(FEATURE_COLOR[abbr])
        
    dfScores = pandas.read_csv(FSScorePath, header = [0, 1])
    header = dfScores.columns.values[columnIndex]
    scores = getScores(dfScores, " + ".join(features), header)
    
    # plot figure
    featureNames, featureColors = [], []
    for i in range(len(features)):
        featureNames += headerGroupMap[features[i]]
        featureColors += headerColorMap[features[i]]
    
    if sortScores:
        tuples = sorted([(scores[i], featureNames[i], featureColors[i]) for i in range(len(scores))], key = lambda x : x[0])
        scores, featureNames, featureColors = [], [], []
        for i in range(len(tuples)):
            scores.append(tuples[i][0])
            featureNames.append(tuples[i][1])
            featureColors.append(tuples[i][2]) 
    print(scores[:10])
    YPos = np.arange(len(featureNames))
    fig, ax = plt.subplots(figsize = (8, 18))
    ax.barh(YPos, scores, color = featureColors, align = "center")
    ax.set_title("Feature Importance Scores " + str(header))
    ax.set_yticks(YPos)
    ax.set_yticklabels(featureNames, fontsize = 8)
    ax.invert_yaxis()
    plt.tight_layout()
    if figPath:
        fig.savefig(figPath)
    else:
        plt.show()
    
if __name__ == "__main__":
    '''
    parser = initializeParser()
    args = parser.parse_args()
    dataPath, FSScorePath, features, column, figPath, sortScores = args.data_path, args.FSScore_path, args.feature_combination, args.column, args.fig_path, args.sort_scores
    '''
    
    '''
    dataPath = os.path.join("..", "Hariri", "Cross-site_data", "compiled_dataset_variables_v4.xlsx")
    FSScorePath = "FSScores.csv"
    features = "LS + PD + FA + DM + SP"
    column = "AB"
    figPath = os.path.join("..", "Figures", "FSScores.png")
    sortScores = True
    '''
    
    makeFSScorePlot(dataPath, FSScorePath, features, column, figPath, sortScores)