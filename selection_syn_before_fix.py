import sys
sys.path.append("../")
sys.path.append("AIF360/")
import numpy as np
from tqdm import tqdm
from fairness_metrics.tot_metrics import TPR,TNR,get_BCR,DIbinary
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from common_utils import compute_metrics
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import random


import os

import pandas as pd

from aif360.datasets import StandardDataset
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

def get_distortion_custom(vold, vnew):
    """Distortion function for the compas dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """
    # Distortion cost
    distort = {}
    distort['two_year_recid'] = pd.DataFrame(
                                {'No recid.':     [0., 2.],
                                'Did recid.':     [2., 0.]},
                                index=['No recid.', 'Did recid.'])
    distort['age_cat'] = pd.DataFrame(
                            {'Less than 25':    [0., 1., 2.],
                            '25 to 45':         [1., 0., 1.],
                            'Greater than 45':  [2., 1., 0.]},
                            index=['Less than 25', '25 to 45', 'Greater than 45'])

    distort['c_charge_degree'] = pd.DataFrame(
                            {'M':   [0., 2.],
                            'F':    [1., 0.]},
                            index=['M', 'F'])
    distort['priors_count'] = pd.DataFrame(
                            {'0':           [0., 1., 2.],
                            '1 to 3':       [1., 0., 1.],
                            'More than 3':  [2., 1., 0.]},
                            index=['0', '1 to 3', 'More than 3'])

    
    distort['score_text'] = pd.DataFrame(
                                {'Low':     [0., 2.],
                                'MediumHigh':     [2., 0.]},
                                index=['Low', 'MediumHigh'])
    distort['sex'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])
    distort['race'] = pd.DataFrame(
                        {0.0:    [0., 2.],
                         1.0:    [2., 0.]},
                         index=[0.0, 1.0])

    total_cost = 0.0
    for k in vold:
        if k in vnew:
            total_cost += distort[k].loc[vnew[k], vold[k]]

    return total_cost




class CustomDataset(StandardDataset):
    """Adult Census Income Dataset.

    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, label_name='y',
                 favorable_classes=['1'],
                 protected_attribute_names=['x_control'],
                 privileged_classes=['0'],
                 instance_weights_name=None,
                 categorical_features=[],
                 features_to_keep=[], features_to_drop=[],
                 na_values=[''], custom_preprocessing=None,
                 df = None,
                 metadata=None):



        super().__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)

def train():
    df = pd.read_csv('syn_train.csv')
    df_test = pd.read_csv('syn_test.csv')



    df_neg = df.loc[df['two_year_recid']==1,:]
    df_neg_priv = df_neg.loc[(df_neg['two_year_recid']==1)&(df_neg['race']==1),:]
    df_neg_unpriv = df_neg.loc[(df_neg['two_year_recid']==1)&(df_neg['race']==0),:]



    df_neg = df.loc[df['two_year_recid']==1,:]
    df_neg_priv = df_neg.loc[(df_neg['two_year_recid']==1)&(df_neg['race']==1),:]
    df_neg_unpriv = df_neg.loc[(df_neg['two_year_recid']==1)&(df_neg['race']==0),:]

    _,df_neg_priv_test =train_test_split(df_neg_priv,test_size=1200,random_state=1)
    _,df_neg_unpriv_test =train_test_split(df_neg_unpriv,test_size=2800,random_state=1)
    df_neg_test = df_neg_priv_test.append(df_neg_unpriv_test)
    print('negative outcome, unpriv')
    print(len(df_neg_unpriv_test.index))

    print('negative outcome, priv')
    print(len(df_neg_priv_test.index))
    df_pos = df.loc[df['two_year_recid']==0,:]
    df_pos_priv = df_pos.loc[(df_pos['two_year_recid']==0)&(df_pos['race']==1),:]
    df_pos_unpriv = df_pos.loc[(df_pos['two_year_recid']==0)&(df_pos['race']==0),:]
    _,df_pos_priv_test =train_test_split(df_pos_priv,test_size=2000,random_state=1)
    _,df_pos_unpriv_test =train_test_split(df_pos_unpriv,test_size=2000,random_state=1)
    df_pos_test = df_pos_priv_test.append(df_pos_unpriv_test)
    df = df_pos_test.append(df_neg_test)
    print('positive outcome, unpriv')
    print(len(df_pos_unpriv_test.index))
    print('positive outcome, priv')
    print(len(df_pos_priv_test.index))

    df_train = df
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    all_protected_attribute_maps = {"race": {0.0: 0, 1.0: 1}}
    D_features = ['race']
    dataset_orig_train = CustomDataset(label_name = 'two_year_recid',favorable_classes = [0],protected_attribute_names=['race'],
                                 privileged_classes=[[1]],categorical_features=['priors_count', 'c_charge_degree','age_cat', 'score_text'],
                                 features_to_keep = ['priors_count', 'c_charge_degree', 'race', 'age_cat', 'score_text'],df=df_train,
                                 metadata={'label_maps': [{1: 'Did recid.', 0: 'No recid.'}],'protected_attribute_maps': [all_protected_attribute_maps[x]
                                    for x in D_features]})

    dataset_orig_vt = CustomDataset(label_name = 'two_year_recid',favorable_classes = [0],protected_attribute_names=['race'],
                                 privileged_classes=[[1]],categorical_features=['priors_count', 'c_charge_degree','age_cat', 'score_text'],
                                 features_to_keep = ['priors_count', 'c_charge_degree', 'race', 'age_cat', 'score_text'],df=df_test,
                                 metadata={'label_maps': [{1: 'Did recid.', 0: 'No recid.'}],'protected_attribute_maps': [all_protected_attribute_maps[x]
                                    for x in D_features]})


    optim_options = {
        "distortion_fun": get_distortion_custom,
        "epsilon": 0.2,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.2, 0.1, 0]
    }

    # dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)


    metric_transf_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)



    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups = unprivileged_groups,
                      privileged_groups = privileged_groups)

    OP = OP.fit(dataset_orig_train)



    dataset_transf_cat_test = OP.transform(dataset_orig_vt, transform_Y = True)
    dataset_transf_cat_test = dataset_orig_vt.align_datasets(dataset_transf_cat_test)


    dataset_transf_cat_train = OP.transform(dataset_orig_train, transform_Y = True)
    dataset_transf_cat_train = dataset_orig_train.align_datasets(dataset_transf_cat_train)


    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_cat_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    print(metric_transf_train.mean_difference())



    scale_transf = StandardScaler()
    X_train = dataset_orig_train.features
    y_train = dataset_orig_train.labels.ravel()
    X_test = scale_transf.fit_transform(dataset_orig_vt.features)


    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_pred = lmod.predict(X_test)
    print('Without resampling')
    print('Accuracy')
    print(accuracy_score(dataset_orig_vt.labels, y_pred))
    dataset_orig_vt_copy = dataset_orig_vt.copy()
    dataset_orig_vt_copy.labels = y_pred
    metric_transf_train = BinaryLabelDatasetMetric(dataset_orig_vt_copy, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    print('p-rule')
    print(metric_transf_train.disparate_impact())
    print('CV')
    print(metric_transf_train.mean_difference())
    print('FPR for unpriv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],0))
    print('FNR for unpriv')
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],0))
    print('FPR for priv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],0))
    print('FNR for priv')
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],0))
    

if __name__ == '__main__':
    train()


