import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
import pandas as pd

absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\..\\'))
result_path = absolute_path + '\\result\\'

y_test_KNN = pd.read_csv(result_path + "KNN\\y_test_5fold.csv", index_col=[0])
y_test_RF = pd.read_csv(result_path + "RF\\y_test_5fold.csv", index_col=[0])
y_test_SVM = pd.read_csv(result_path + "SVM\\y_test_5fold.csv", index_col=[0])

y_pred_KNN = pd.read_csv(result_path + "KNN\\y_pred_5fold.csv", index_col=[0])
y_pred_RF = pd.read_csv(result_path + "RF\\y_pred_5fold.csv", index_col=[0])
y_pred_SVM = pd.read_csv(result_path + "SVM\\y_pred_5fold.csv", index_col=[0])

fpr1, tpr1, _1 = roc_curve(y_true=y_test_KNN, y_score=y_pred_KNN, pos_label=1)
fpr2, tpr2, _2 = roc_curve(y_true=y_test_RF, y_score=y_pred_RF, pos_label=1)
fpr3, tpr3, _3 = roc_curve(y_true=y_test_SVM, y_score=y_pred_SVM, pos_label=1)

roc_auc1= auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)


plt.figure()
lw = 2
plt.plot(fpr1,tpr1, color="darkorange",lw=lw, linestyle="--", label="KNN (area = %0.4f)" % roc_auc1)
plt.plot(fpr2,tpr2, color="navy",lw=lw, linestyle=":", label="RF (area = %0.4f)" % roc_auc2)
plt.plot(fpr3,tpr3, color="red",lw=lw, linestyle="-.", label="SVM (area = %0.4f)" % roc_auc3)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve of classifiers for 5 fold")
plt.legend(loc="lower right")
plt.show()
# plt.savefig(fname=result_path + "\\ROC_5fold.eps")