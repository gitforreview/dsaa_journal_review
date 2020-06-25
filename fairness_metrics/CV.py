from DIbinary import DIbinary

def CV(pred,sensitive,unprotected_vals,positive_pred):
    unprotected_pos_percent, protected_pos_percent = DIbinary(pred, sensitive, unprotected_vals,positive_pred)
    cv = unprotected_pos_percent -protected_pos_percent
    return 1 - cv
