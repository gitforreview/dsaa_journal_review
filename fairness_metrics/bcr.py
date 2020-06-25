import tnr
import tpr
# acutal is the list of actual labels, pred is the list of predicted labels.
# positive_pred is the favorable result in prediction task. e.g. get approved for a loan
def get_BCR(actual,pred,positive_pred):
    tpr_val = TPR(actual,pred,positive_pred)
    tnr_val = TNR(actual,pred,positive_pred)
    bcr = (tpr_val + tnr_val) / 2.0
    return bcr
