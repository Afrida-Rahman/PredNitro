import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
import pandas as pd

absolute_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..\\..\\'))
result_path = absolute_path + '\\result\\KNN\\'

y_test_5fold = pd.read_csv(result_path + "y_test_5fold.csv", index_col=[0])
y_test_10fold = pd.read_csv(result_path + "y_test_10fold.csv", index_col=[0])
y_test_jackknife = pd.read_csv(result_path + "y_test_jackknife.csv", index_col=[0])

y_pred_5fold = pd.read_csv(result_path + "y_pred_5fold.csv", index_col=[0])
y_pred_10fold = pd.read_csv(result_path + "y_pred_10fold.csv", index_col=[0])
y_pred_jackknife = pd.read_csv(result_path + "y_pred_jackknife.csv", index_col=[0])

fpr1, tpr1, _1 = roc_curve(y_true=y_test_5fold, y_score=y_pred_5fold, pos_label=1)
fpr2, tpr2, _2 = roc_curve(y_true=y_test_10fold, y_score=y_pred_10fold, pos_label=1)
fpr3, tpr3, _3 = roc_curve(y_true=y_test_jackknife, y_score=y_pred_jackknife, pos_label=1)

roc_auc1= auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)


plt.figure()
lw = 2
plt.plot(fpr1,tpr1, color="darkorange",lw=lw, label="ROC curve (area = %0.2f)" % roc_auc1)
plt.plot([0, 1], [0, 0], color="navy", lw=lw, linestyle="--")
plt.plot(fpr2,tpr2, color="darkorange",lw=lw, label="ROC curve (area = %0.2f)" % roc_auc2)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="-")
plt.plot(fpr3,tpr3, color="darkorange",lw=lw, label="ROC curve (area = %0.2f)" % roc_auc3)
plt.plot([0, 1], [1, 0], color="navy", lw=lw, linestyle="-.")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
# plt.savefig(fname=result_path + "\\ROC.eps")