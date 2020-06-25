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
