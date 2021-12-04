import numpy as np
import pandas as pd
import math
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os


absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\'))
feature_path = absolute_path + '\\feature_extract\\'
cross_val_path = absolute_path + '\\fold_index\\'
result_path = absolute_path + '\\result\\'

dataset= pd.read_csv(feature_path+'\\Prob_Feature.csv', header = None)
X = dataset.iloc[:, :].values
X_train=X

y= []
for i in range(1044):
    y.append(1);
for i in range(7669):
    y.append(-1);

y= np.array(y, dtype=np.int64)
y_train= y

wap = 8713/(2*1044)
wan = 8713/(2*7669)
weight = {-1:wan, 1:wap}  

classifier= SVC(C=1, kernel='rbf', gamma=1/X_train.shape[1], class_weight = weight, cache_size=500,  random_state = 0)
#classifier = SVC()
train_index = pd.read_csv(cross_val_path+'\\train_index_LOO.csv',header=None)
test_index = pd.read_csv(cross_val_path+'\\test_index_LOO.csv',header=None)

bp=[]
results = []
i = 1
y_test_fold=[]
y_pred_fold=[]
y_pred_score_fold=[]
for fold in range(train_index.shape[1]):  #### flaws here ###
    train_ind = pd.DataFrame(train_index.iloc[:,fold].values).dropna()
    train_ind = np.array(train_ind, dtype=np.int64)
    train_ind=np.reshape(train_ind,(len(train_ind,)))
    
    test_ind = pd.DataFrame(test_index.iloc[:,fold].values).dropna()
    test_ind = np.array(test_ind, dtype=np.int64)
    test_ind=np.reshape(test_ind,(len(test_ind,)))
    
    X_train_split = X_train[train_ind]
    y_train_split = y_train[train_ind]
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
    print(i)
    i+=1


y_test_fold = np.array(y_test_fold)
y_pred_fold = np.array(y_pred_fold)
y_pred_score_fold = np.array(y_pred_score_fold)

y_test_fold = np.reshape(y_test_fold, (len(y_test_fold),1))
y_pred_fold = np.reshape(y_pred_fold, (len(y_pred_fold),1))
y_pred_score_fold = np.reshape(y_pred_score_fold, (len(y_pred_score_fold),1))

y_test_fold = pd.DataFrame(y_test_fold)
y_pred_fold = pd.DataFrame(y_pred_fold)
y_pred_score_fold = pd.DataFrame(y_pred_score_fold)

acc = accuracy_score(y_true = y_test_fold, y_pred = y_pred_fold)
mcc = matthews_corrcoef(y_true = y_test_fold, y_pred = y_pred_fold, sample_weight=None)
precision = sklearn.metrics.precision_score(y_true = y_test_fold, y_pred = y_pred_fold)
sp = sklearn.metrics.recall_score(y_true = y_test_fold, y_pred = y_pred_fold, pos_label =-1) 
sn = sklearn.metrics.recall_score(y_true = y_test_fold, y_pred = y_pred_fold, pos_label =1) 
auc = sklearn.metrics.roc_auc_score(y_true = y_test_fold, y_score = y_pred_score_fold)

result = [sp, sn, precision, acc, mcc, auc]
result = np.array(result)
result = np.reshape(result, (1,6))
result = pd.DataFrame(result)
print(result)
result.to_csv(result_path+'\\prob_jacknife_libsvm.csv', header = ["Sp", "Sn", "pre","ACC","MCC","AUC"])

