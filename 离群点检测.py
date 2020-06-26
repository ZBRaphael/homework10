#!/usr/bin/env python
# coding: utf-8

# # 离群点检测

# In[1]:


# -*- coding: utf-8 -*-
"""Example of using kNN for outlier detection
"""

from __future__ import division
from __future__ import print_function


import os
import sys
import pandas as pd

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.model_selection import train_test_split


# ### 对于单个csv处理的过程
# 
# 将数据集剔除三个无用的属性，'point.id', 'motherset', 'origin'

# In[2]:


df = pd.read_csv("./wine/benchmarks/wine_benchmark_0001.csv")
columns = df.columns
df = df[columns].fillna('nan')


# In[3]:


columns


# In[4]:


data = df.drop(columns = ['point.id', 'motherset', 'origin'])


# In[5]:


class_mapping = {"anomaly":1, "nominal":0}


# In[6]:


data['ground.truth'] = data['ground.truth'].map(class_mapping)


# 将数据集中除了ground.truth的属性当作X，ground.truth属性当作y

# In[7]:


data.head()


# In[8]:


class_mapping = {"anomaly":1, "nominal":0}


# In[9]:


y = data['ground.truth']


# In[10]:


x = data.drop('ground.truth',axis=1)


# 数据集中使用8:2划分训练和测试集

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=28)


# 使用Knn算法对数据离群点进行判断

# In[12]:


clf_name = 'KNN'
clf = KNN()
clf.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

# evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)


# ### 使用多个算法对数据集进行处理
# 
# - ABOD 
# - CBLOF 
# - LOF  
# - HOBS  
# - IForest 
# - KNN  
# - AKNN 

# 异常检测问题往往是没有标签的，训练数据中并未标出哪些是异常点，因此必须使用无监督学习,在进行检测时使用标签进行对比

# In[16]:


import numpy as np
random_state = np.random.RandomState(42)
outliers_fraction = 0.05
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
outliers_fraction = 0.05
# Define seven outlier detection tools to be compared
classifiers = {
#         'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}
for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, y_test, y_test_scores)


# ### 对数据集中csv进行遍历并使用多个算法进行检测
# 
# 在本地进行执行，所以只在后面统计结果

# In[ ]:


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

file_list = []
total_roc = []
total_prn = []
count = 0
for home, dirs, files in os.walk("./skin/benchmarks"):
    for filename in files:
        fullname = os.path.join(home, filename)
        file_list.append(fullname)
for file_csv in file_list:
    
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
            p_roc.append(roc)
        except:
            p_prn.append(-1)
            p_roc.append(-1)

    total_prn.append(p_prn)
    total_roc.append(p_roc)    
    count += 1
        
total_prn = json.dumps(total_prn)
total_roc = json.dumps(total_roc)
a = open("skin_prc_list.txt", "w",encoding='UTF-8')
a.write(total_prn)
a.close()
a = open("skin_roc_list.txt", "w",encoding='UTF-8')
a.write(total_roc)
a.close()


# In[126]:


import json
b = open("./wine_roc_list.txt", "r",encoding='UTF-8')
out = b.read()
out =  json.loads(out)


# In[131]:


len(out)


# ### 显示示例结果

# In[128]:


out[0:3]


# ### 计算wine数据集平均ROC和PRN值
# 
# 其中-1表示该算法在数据集上运行时出错，在计算平均值时剔除

# In[129]:


total_roc = []
for ip in range(0,7): 
    total = 0
    count = 0
    for r in out:
        if r[ip] != -1:
            total += r[ip]
            count += 1
    total_roc.append(total/count)


# In[130]:


total_roc


# ABOD = 0.6210063073394493,
# CBLOF = 0.6189723801065717,
# LOF = 0.6460626998223798,
# HOBS = 0.5912920959147424,
# IForest = 0.687247513321493,
# KNN = 0.6485466252220252,
# AKNN = 0.6578079040852582
# 
# 

# In[132]:


import json
b = open("./wine_prn_list.txt", "r",encoding='UTF-8')
out = b.read()
out =  json.loads(out)


# In[133]:


total_prn = []
for ip in range(0,7): 
    total = 0
    count = 0
    for r in out:
        if r[ip] != -1:
            total += r[ip]
            count += 1
    total_prn.append(total/count)


# In[134]:


len(out)


# In[135]:


total_prn


# ABOD = 0.17593841743119282,
# CBLOF = 0.16324582593250422,
# LOF = 0.17491678507992875,
# HOBS = 0.12832815275310824,
# IForest = 0.20096989342806384,
# KNN = 0.17297602131438716,
# AKNN = 0.17796243339253962
# 
# 通过对比在wine数据集中，使用IForest算法训练和检测的准确率最高

# ### 计算skin数据集平均ROC和PRN值
# 
# 其中-1表示该算法在数据集上运行时出错，在计算平均值时剔除

# In[136]:


b = open("./skin_roc_list.txt", "r",encoding='UTF-8')
out = b.read()
out =  json.loads(out)


# In[137]:


total_roc = []
for ip in range(0,7): 
    total = 0
    count = 0
    for r in out:
        if r[ip] != -1:
            total += r[ip]
            count += 1
    total_roc.append(total/count)


# In[138]:


len(out)


# In[139]:


total_roc


# In[142]:


b = open("./skin_prn_list.txt", "r",encoding='UTF-8')
out = b.read()
out =  json.loads(out)
total_prn = []
for ip in range(0,7): 
    total = 0
    count = 0
    for r in out:
        if r[ip] != -1:
            total += r[ip]
            count += 1
    total_prn.append(total/count)


# In[143]:


total_prn


# 通过对比在skin数据集中，使用HOBS算法训练和检测的准确率最高,另外使用IForest算法的准确了也是相当不错
