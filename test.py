# -*- coding: utf-8 -*-
"""Example of using kNN for outlier detection
"""

from __future__ import division, print_function

import json
import os
import sys

import numpy as np
import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.data import evaluate_print, generate_data
from pyod.utils.example import visualize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pyod.utils import precision_n_scores

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
def fun(dir_path):
    file_list = []
    total_roc = []
    total_prn = []
    count = 0
    for home, dirs, files in os.walk("./"+dir_path+"/benchmarks"):
        for filename in files:
            fullname = os.path.join(home, filename)
            file_list.append(fullname)cb
    for file_csv in file_list:

        # if count == 2:
        #     break
        
        df = pd.read_csv(file_csv)
        columns = df.columns
        # df = df[columns].fillna('nan')

        data = df.drop(columns = ['point.id', 'motherset', 'origin'])

        class_mapping = {"anomaly":1, "nominal":0}
        data['ground.truth'] = data['ground.truth'].map(class_mapping)
        class_mapping = {"anomaly":1, "nominal":0}

        y = data['ground.truth']

        x = data.drop('ground.truth',axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=28)

        random_state = np.random.RandomState(42)
        outliers_fraction = 0.05
        # Define seven outlier detection tools to be compared
        classifiers = {
                'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
                'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
                'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
                'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
                'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
                'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
                'Average KNN': KNN(method='mean',contamination=outliers_fraction)
        }
        p_prn = []
        p_roc = []
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            try:
                clf.fit(X_train)

                # get the prediction labels and outlier scores of the training data
                y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
                y_train_scores = clf.decision_scores_  # raw outlier scores

                # get the prediction on the test data
                y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
                y_test_scores = clf.decision_function(X_test)  # outlier scores

                # evaluate and print the results
                print(str(count)+"is analysing")
                print("\nOn Training Data:")
        
                evaluate_print(clf_name, y_train, y_train_scores)
                print("\nOn Test Data:")
                evaluate_print(clf_name, y_test, y_test_scores)
                roc=np.round(roc_auc_score(y_train, y_train_scores), decimals=4),
                prn=np.round(precision_n_scores(y_test, y_test_scores), decimals=4)

                p_prn.append(prn)
                p_roc.append(roc[0])
            except:
                p_prn.append(-1)
                p_roc.append(-1)

        total_prn.append(p_prn)
        total_roc.append(p_roc)    
        count += 1
            
    total_prn = json.dumps(total_prn)
    total_roc = json.dumps(total_roc)
    a = open(dir_path+"_prn_list.txt", "w",encoding='UTF-8')
    a.write(total_prn)
    a.close()
    a = open(dir_path+"_roc_list.txt", "w",encoding='UTF-8')
    a.write(total_roc)
    a.close()
if __name__ == "__main__":
    fun("skin")
    fun("wine")
