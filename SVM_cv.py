#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:34:00 2018

@author: zhuoqinyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:05:59 2017

@author: zhuoqinyu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import sklearn as sk
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#y_tr = pd.read_csv('deeptox_descriptor/tox21_labels_train.csv.gz', index_col=0, compression="gzip")
x_tr_dense = pd.read_csv('../deeptox_descriptor/tox21_dense_train.csv.gz', index_col=0, compression="gzip")

#x_tr_sparse = io.mmread('tox21_sparse_train.mtx.gz').tocsc()
simdf=pd.read_pickle('../oned_similarity/sim_target_df.pkl')
#combine two dataframe
targets_df=simdf.loc[:,"nr-ahr":"sr-p53"]
x_df=pd.concat([x_tr_dense,targets_df],axis=1, join='inner')

tasks=targets_df.columns.tolist()
# Build a random forest model for all twelve assays

for task in tasks:
    print (task)
    new_x_df=x_df[x_df[task]!="NAN"]
    x_task=new_x_df.ix[:,0:801]
    target=new_x_df[task].astype('int32')
    train,test,y_train,y_test=train_test_split(x_task,target,train_size=0.7,random_state=1)
    scaler=sk.preprocessing.StandardScaler().fit(train)
    train_scaled=scaler.transform(train)
    test_scaled=scaler.transform(test)
    param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                 'C': [1, 10, 100],'class_weight':["balanced"]},
                {'kernel': ['linear'], 'C': [1, 10, 100],'class_weight':["balanced"]}]
 
    CV_clf = GridSearchCV(svm.SVC(), param_grid=param_grid, cv= 10, scoring='roc_auc')
    CV_clf.fit(x_train_pca, y_train)
    print('best_cv_parameter')
    print(CV_clf.best_params_)
    
    #optimize rf alorithm
    seed=123
    clf = svm.SVC(**CV_clf.best_params_)
    clf.fit(train_scaled, y_train)
    pred=clf.predict(test_scaled)
    p_te = clf.predict_proba(test_scaled)
    auc_te = roc_auc_score(y_test,p_te[:,1])
    print('AUC:')
    print("%15s: %3.5f" % (task, auc_te))
