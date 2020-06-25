from sklearn.metrics import recall_score
# acutal is the list of actual labels, pred is the list of predicted labels.
# positive_pred is the favorable result in prediction task. e.g. get approved for a loan
def TPR(actual,pred,positive_pred):
    return recall_score(actual,pred,pos_label = positive_pred,average = 'binary')

TPR(y_test,y_pred,1)
