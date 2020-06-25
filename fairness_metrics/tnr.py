from sklearn.metrics import confusion_matrix

# acutal is the list of actual labels, pred is the list of predicted labels.
# positive_pred is the favorable result in prediction task. e.g. get approved for a loan
def TNR(actual,pred,positive_pred):
    classes = list(set(actual))
    matrix = confusion_matrix(actual, pred, labels=classes)
    TN = 0.0
    allN = 0.0
    for i in range(0, len(classes)):
        trueval = classes[i]
        if trueval == positive_pred:
            continue
        for j in range(0, len(classes)):
            allN += matrix[i][j]
            predval = classes[j]
            if trueval == predval:
                TN += matrix[i][j]
    if allN == 0.0:
        return 1.0
    print(allN)
    return TN / allN
