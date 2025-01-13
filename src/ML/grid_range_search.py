from ML.common import *

def SVM(X, Y, processes, output_dir):
    splitter = LeaveOneOut()
    
    # rbf
    grid = {"C" : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
            "gamma" : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]}
    splits = splitter.split(X, Y)
    clf = GridSearchCV(estimator = SVC(kernel = "rbf"), param_grid = grid, scoring = "accuracy", cv = splits, refit = False, n_jobs = processes, return_train_score = False)
    clf.fit(X, Y)
    dfOut = pandas.DataFrame(clf.cv_results_)
    dfOut.to_csv(os.path.join(output_dir, "RBFSVM_Grid_Results.csv"), index = False)
    
    # linear
    grid = {"C" : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]}
    splits = splitter.split(X, Y)
    clf = GridSearchCV(estimator = SVC(kernel = "linear"), param_grid = grid, scoring = "accuracy", cv = splits, refit = False, n_jobs = processes, return_train_score = False)
    clf.fit(X, Y)
    dfOut = pandas.DataFrame(clf.cv_results_)
    dfOut.to_csv(os.path.join(output_dir, "LinearSVM_Grid_Results.csv"), index = False)

def RF(X, Y, processes, output_dir):
    splitter = LeaveOneOut()
    
    grid = {"criterion" : ["gini", "entropy"],
            "max_depth" : [4, 8, 12, 16, 20, 24, 28, 32, 36],
            "max_features" : ["log2", "sqrt", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    splits = splitter.split(X, Y)
    clf = GridSearchCV(estimator = RFC(n_estimators = 800, random_state = 4), param_grid = grid, scoring = "accuracy", cv = splits, refit = False, n_jobs = processes, return_train_score = False)
    clf.fit(X, Y)
    dfOut = pandas.DataFrame(clf.cv_results_)
    dfOut.to_csv(os.path.join(output_dir, "RF_Grid_Results.csv"), index = False)

def AdaBoost(X, Y, processes, output_dir):
    splitter = LeaveOneOut()
    splits = splitter.split(X, Y)
    
    grid = {"n_estimators" : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "learning_rate" : [1e-3, 1e-2, 1e-1, 1]}
    '''
    grid = {"n_estimators" : [100 + 50 * i for i in range(19)],
            "learning_rate" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    '''
    clf = GridSearchCV(estimator = ABC(base_estimator = DTC(max_depth = 1), random_state = 4), param_grid = grid, scoring = "accuracy", cv = splits, refit = False, n_jobs = processes, return_train_score = False)
    clf.fit(X, Y)
    dfOut = pandas.DataFrame(clf.cv_results_)
    dfOut.to_csv(os.path.join(output_dir, "AdaBoost_Grid_Results.csv"), index = False)
    
