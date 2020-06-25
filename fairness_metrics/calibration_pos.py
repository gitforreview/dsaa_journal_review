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

