# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import math
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.metrics import specificity_score
import csv
import imblearn
import os

absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\'))
feature_path = absolute_path + '\\feature_extract\\'
cross_val_path = absolute_path + '\\fold_index\\'
result_path = absolute_path + '\\result\\'

dataset = pd.read_csv(feature_path+'\\Prob_Feature.csv', header = None)
X = dataset.iloc[:, :].values
y= []
for i in range(1151):
    y.append(1);
for i in range(7666):
    y.append(-1);

y= np.array(y, dtype=np.int64)
y_train= y
X_train=X


train_index = pd.read_csv(cross_val_path+'\\train_index_10fold.csv',header=None)
test_index = pd.read_csv(cross_val_path+'\\test_index_10fold.csv',header=None)


bp=[]
results = []
i = 1
y_test_fold=[]
y_pred_fold=[]
y_pred_score_fold=[]
 
C=1
gamma=1/X_train.shape[1]

wap = 8817/(2*1151)
wan = 8817/(2*7666)
weight = {-1:wan, 1:wap}  

time=0
for fold in range(train_index.shape[1]):  
    
    train_ind = pd.DataFrame(train_index.iloc[:,fold].values).dropna()
    train_ind = np.array(train_ind, dtype=np.int64)
    train_ind=np.reshape(train_ind,(len(train_ind,)))
    
    test_ind = pd.DataFrame(test_index.iloc[:,fold].values).dropna()
    test_ind = np.array(test_ind, dtype=np.int64)
    test_ind=np.reshape(test_ind,(len(test_ind,)))
    
    if fold%10==0:
        time=int(fold/10)
    
    X_train_split = X_train[train_ind]
    y_train_split = y_train[train_ind]
   
    classifier = SVC(C=C, kernel='rbf', gamma=gamma, class_weight=weight, cache_size=500,  random_state = 0)
    classifier.fit(X_train_split, y_train_split)    
    X_test_split = X_train[test_ind]
    y_test_split = y_train[test_ind]
    y_pred = classifier.predict(X_test_split)
    y_pred_score = classifier.decision_function(X_test_split)
    y_test_split = y_test_split.reshape(y_test_split.shape[0],-1)
    y_pred_f = y_pred.reshape(y_pred.shape[0],-1)
    y_pred_score_f = y_pred_score.reshape(y_pred_score.shape[0],-1)

    y_test_fold.append(y_test_split)
    y_pred_fold.append(y_pred_f)
    y_pred_score_fold.append(y_pred_score_f)
    
    if i % 10 == 0:
        x=0
        y_test_time = np.concatenate([y_test_fold[x] for x in range(10)])
        y_test_fold = []
        x=0
        y_pred_time = np.concatenate([y_pred_fold[x] for x in range(10)])
        y_pred_fold = []
        x=0
        y_pred_score_time = np.concatenate([y_pred_score_fold[x] for x in range(10)])
        y_pred_score_fold = []
        
        acc = accuracy_score(y_true = y_test_time, y_pred = y_pred_time)
        mcc = matthews_corrcoef(y_true = y_test_time, y_pred = y_pred_time, sample_weight=None)
        sp=imblearn.metrics.specificity_score(y_true=y_test_time, y_pred=y_pred_time, labels=None, pos_label=1, average='binary', sample_weight=None)
        sn=imblearn.metrics.sensitivity_score(y_true=y_test_time, y_pred=y_pred_time, labels=None, pos_label=1, average='binary', sample_weight=None)
        auc = sklearn.metrics.roc_auc_score(y_true = y_test_time, y_score = y_pred_score_time)
        
        curr_res = []
        curr_res.append(acc)
        curr_res.append(mcc)
        curr_res.append(sp)
        curr_res.append(sn)
        curr_res.append(auc)
        results.append(curr_res)
        print(i)
    i+=1


results.insert(0, ["Accuracy", "MCC","sp","sn","auc"])
with open(result_path+'\\prob_libsvm_10fold.csv', 'w', newline="") as myfile2:
    wr = csv.writer(myfile2)   
    wr.writerows(results)
    
