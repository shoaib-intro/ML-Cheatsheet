#--------------------- Performance Measures 
#only supported with categorical variables 

from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score, cohen_kappa_score, roc_auc_score
# accuracy: tp + tn / p + n
accuracy = (y_true,y_pred)
print('Accuracry Score {%0.3f}'.format(accuracy))

# precison: tp / tp + fp
precision = precision_score(y_true,y_pred)
print('precion {%0.3f}'.format(precision))

# recall: tp / tp + fn
recall = recall_score(y_true,y_pred)
print('recall {%0.3f}'.format(recall))

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true, y_pred)
print('F1 Score {%0.3f}'.format(f1))

# kappa
kappa = cohen_kappa_score(y_true, y_pred)
print('Cohen Kappa {%0.3f}'.format(kappa))

# ROC AUC
roc_auc = roc_auc_score(y_true, y_pred)
print('ROC AUC {%0.3f}'.format(roc_auc))
