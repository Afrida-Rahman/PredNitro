from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import imblearn
import os
import joblib


absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\..\\'))
feature_path_training = absolute_path + '\\training\\PredNTS Dataset\\feature_extract\\'
feature_path_testing = absolute_path + '\\test\\feature_extract\\'
result_path = absolute_path + '\\test\\result\\'


training_dataset = pd.read_csv(feature_path_training+'\\Prob_Feature.csv', header = None)
test_dataset = pd.read_csv(feature_path_testing+'\\Prob_Feature.csv', header = None)

X_train = training_dataset.iloc[:, :].values
X_test = test_dataset.iloc[:, :].values



y_train= []
for i in range(1191):
    y_train.append(1)
for i in range(1191):
    y_train.append(-1)

y_train= np.array(y_train, dtype=np.int64)

y_test= []
for i in range(203):
    y_test.append(1)
for i in range(1022):
    y_test.append(-1)

y_test= np.array(y_test, dtype=np.int64)
classifier = SVC(gamma='auto')
print('Starting to train!')
classifier.fit(X_train, y_train)
print('Training finished! Saving model weights...')

####  saving models  ####
model = result_path + '\\final_model.sav'
joblib.dump(classifier, model)
print('Model saved! Now starting test...')

### Loading models ###
model_loaded = result_path + '\\final_model.sav'
classifier_loaded = joblib.load(model_loaded)

y_pred = classifier_loaded.predict(X_test)
y_pred_score = classifier_loaded.decision_function(X_test)
print('Test finished!')

    
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
mcc = matthews_corrcoef(y_true = y_test, y_pred = y_pred, sample_weight=None)
sp=imblearn.metrics.specificity_score(y_true=y_test, y_pred=y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
sn=imblearn.metrics.sensitivity_score(y_true=y_test, y_pred=y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
auc = sklearn.metrics.roc_auc_score(y_true = y_test, y_score = y_pred_score)

results = np.array([acc, mcc, sp, sn, auc])
results = pd.DataFrame(results)
results = results.T
print(results)
results.to_csv(result_path+'\\ind-test-result-libsvm.csv', header=['ACC','MCC','Sp','Sn','AUC'], index=None)

y_test_all_fold = pd.DataFrame(np.array(y_test).flatten())
y_pred_score_all_fold = pd.DataFrame(np.array(y_pred_score).flatten())
y_test_all_fold.to_csv(result_path+"y_test.csv")
y_pred_score_all_fold.to_csv(result_path+"y_pred_score.csv")
