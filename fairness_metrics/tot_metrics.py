# pred is the list of predicted labels, sensitive is the list of labels for sensitive attribute
# unprotected_vals is the label for unprotected group. e.g. 1 means unprivilege, 0 means privilege
# positive_pred is the favorable result in prediction task. e.g. get approved for a loan
def DIbinary(pred,sensitive,unprotected_vals,positive_pred):
    unprotected_positive = 0.0
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0
    for i in range(0, len(pred)):
        protected_val = sensitive[i]
        predicted_val = pred[i]
        # when someone is in unprotected group
        if protected_val in unprotected_vals:
            if int(predicted_val) == int(positive_pred):
                unprotected_positive += 1
            else:
                unprotected_negative += 1
        # the person is in protected group ie male = 0
        else:
            if int(predicted_val) == int(positive_pred):
                protected_positive += 1
            else:
                protected_negative += 1
    protected_pos_percent = 0.0
    if protected_positive + protected_negative > 0:
        protected_pos_percent = protected_positive / (protected_positive + protected_negative)
    unprotected_pos_percent = 0.0
    if unprotected_positive + unprotected_negative > 0:
        unprotected_pos_percent = unprotected_positive /  \
            (unprotected_positive + unprotected_negative)
    return unprotected_pos_percent, protected_pos_percent



def CV(pred,sensitive,unprotected_vals,positive_pred):
    protected_pos_percent, unprotected_pos_percent = DIbinary(pred, sensitive, unprotected_vals,positive_pred)
    cv = unprotected_pos_percent -protected_pos_percent
    return 1 - cv

# acutal is the list of actual labels, pred is the list of predicted labels.
# sensitive is the column of sensitive attribute, target_group is s in S = s
# positive_pred is the favorable result in prediction task. e.g. get approved for a loan
def calibration_pos(actual,pred,sensitive,target_group,positive_pred):
    tot_pred_pos = 0
    act_pos = 0
    for act, pred_val, sens in zip(actual, pred,sensitive):
        # the case S != s
         if sens != target_group:
             continue
         else:
             # Yhat = 1
             if pred_val == positive_pred:
                 tot_pred_pos += 1
                 # the case both Yhat = 1 and Y = 1
                 if act == positive_pred:
                     act_pos +=1
    if act_pos == 0 and tot_pred_pos ==0:
        return 1
    if tot_pred_pos == 0:
        return 0
    return act_pos/tot_pred_pos

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
    return TN / allN


from sklearn.metrics import recall_score
# acutal is the list of actual labels, pred is the list of predicted labels.
# positive_pred is the favorable result in prediction task. e.g. get approved for a loan
def TPR(actual,pred,positive_pred):
    return recall_score(actual,pred,pos_label = positive_pred,average = 'binary')


# acutal is the list of actual labels, pred is the list of predicted labels.
# positive_pred is the favorable result in prediction task. e.g. get approved for a loan
def get_BCR(actual,pred,positive_pred):
    tpr_val = TPR(actual,pred,positive_pred)
    tnr_val = TNR(actual,pred,positive_pred)
    bcr = (tpr_val + tnr_val) / 2.0
    return bcr
