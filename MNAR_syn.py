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

def get_distortion_syn(vold, vnew):
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
                            {'0':           [0., 1., 2.,100.],
                            '1 to 3':       [1., 0., 1.,100.],
                            'More than 3':  [2., 1., 0.,100.],
                            'missing':      [0., 0., 0.,1.]},
                            index=['0', '1 to 3', 'More than 3','missing'])

    
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

        
np.random.seed(10)
def quantizePrior1(x):
    if x <=0:
        return 0
    elif 1<=x<=3:
        return 1
    else:
        return 2
def quantizeLOS(x):
    if x<= 7:
        return 0
    if 8<x<=93:
        return 1
    else:
        return 2
def group_race(x):
    if x == "Caucasian":
        return 1.0
    else:
        return 0.0

filepath = 'AIF360/aif360/data/raw/compas/compas-scores-two-years.csv'
df = pd.read_csv(filepath, index_col='id', na_values=[])

df['age_cat'] =df['age_cat'].replace('Greater than 45',2)
df['age_cat'] =df['age_cat'].replace('25 - 45',1)
df['age_cat'] =df['age_cat'].replace('Less than 25',0)
df['score_text'] = df['score_text'].replace('High',1)
df['score_text'] = df['score_text'].replace('Medium',1)
df['score_text'] = df['score_text'].replace('Low',0)
df['priors_count'] = df['priors_count'].apply(lambda x: quantizePrior1(x))
df['length_of_stay'] = (pd.to_datetime(df['c_jail_out'])-
                        pd.to_datetime(df['c_jail_in'])).apply(
                                                lambda x: x.days)
df['length_of_stay'] = df['length_of_stay'].apply(lambda x: quantizeLOS(x))
df = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]
df['c_charge_degree'] = df['c_charge_degree'].replace({'F':0,'M':1})

df1 = df[['priors_count','c_charge_degree','race','age_cat','score_text','two_year_recid']]
feature_list=[]
for index, row in df1.iterrows():
    feature_list.append('\t'.join(row.astype(str).to_list()))
df1['feature_list']=feature_list
df3 = df1.groupby('feature_list').count()/len(df1.index)

df2 = pd.DataFrame()
df2['feature_list'] = list(df3.index)
df2['prob_list'] = list(df3.priors_count)
for index, row in df2.iterrows():
    if row['feature_list'][0]=='0' and row['feature_list'][-1]=='1' and 'African' in row['feature_list']:
        row['prob_list'] = row['prob_list']*10

    elif row['feature_list'][0]=='0' and row['feature_list'][-1]=='1':
        row['prob_list'] = row['prob_list']*7
    elif row['feature_list'][0]=='2' and row['feature_list'][-1]=='0':
        row['prob_list'] = row['prob_list']*7
prob_list = list(df2.prob_list)

df_new = pd.DataFrame()
rng = np.random.default_rng()
prob_list=np.array(prob_list)
prob_list = prob_list/prob_list.sum()
feature_list = rng.choice(list(df2.feature_list),len(df1.index),p=prob_list)
var_list =['priors_count','c_charge_degree','race','age_cat','score_text','two_year_recid']
for i in var_list:
    vars()[i] = []

for i in feature_list:
    tmp = i.split('\t')
    for j in range(len(var_list)):
        vars()[var_list[j]].append(tmp[j])
    
for i in var_list:
    df_new[i] = vars()[i]
    
df = df_new
df1 = df[['priors_count','c_charge_degree','race','age_cat','score_text','two_year_recid']]


tot = []
for index, row in df1.iterrows():
    result = ''
    for j in df1.columns:
        result = result+str(row[j])
    tot.append(result)
df['tmp_feature'] = tot
df['mis_prob'] = 0
for i in df['tmp_feature'].unique():
    if 'African' in i and i[0]=='0' and i[-1]=='0':
        df.loc[df['tmp_feature']==i,'mis_prob'] = 0.8
    elif 'African' not in i and i[0]=='0':
        df.loc[df['tmp_feature']==i,'mis_prob'] = 0.2
    else:
        df.loc[df['tmp_feature']==i,'mis_prob'] = 0.1
new_label=[]
for i,j in zip(df['mis_prob'],df['priors_count']):
    if np.random.binomial(1, i, 1)[0]==1:
        new_label.append(3)
    else:
        new_label.append(j)
df['priors_count'] = new_label
print('Total number of missing values')
print(len(df.loc[df['priors_count']==3,:].index))
print('Total number of observations')
print(len(df.index))
df['priors_count'] = df['priors_count'].astype(int)
df['score_text'] = df['score_text'].astype(int)
df['age_cat'] = df['age_cat'].astype(int)
df['score_text'] = df['score_text'].astype(int)
df['c_charge_degree'] = df['c_charge_degree'].astype(int)
df['two_year_recid'] = df['two_year_recid'].astype(int)

df['c_charge_degree'] = df['c_charge_degree'].replace({0:'F',1:'M'})

def quantizePrior(x):
    if x ==0:
        return '0'
    elif x==1:
        return '1 to 3'
    elif x==2:
        return 'More than 3'
    else:
        return 'missing'
# Quantize length of stay
def quantizeLOS(x):
    if x==0:
        return '<week'
    if x==1:
        return '<3months'
    else:
        return '>3 months'

# Quantize length of stay
def adjustAge(x):
    if x ==0:
        return '25 to 45'
    elif x ==1:
        return 'Greater than 45'
    elif x ==2:
        return 'Less than 25'
def quantizeScore(x):
    if x==1:
        return 'MediumHigh'
    else:
        return 'Low'
    


def group_race(x):
    if x == "Caucasian":
        return 1.0
    else:
        return 0.0

df['priors_count'] = df['priors_count'].apply(lambda x: quantizePrior(x))
df['score_text'] = df['score_text'].apply(lambda x: quantizeScore(x))
df['age_cat'] = df['age_cat'].apply(lambda x: adjustAge(x))
# Recode sex and race
df['race'] = df['race'].apply(lambda x: group_race(x))
df['race'] = df['race'].astype(int)

df['two_year_recid'] = df['two_year_recid'].astype(int)

df = df[['priors_count', 'c_charge_degree', 'race', 'age_cat', 'score_text','two_year_recid']]

df_train,df_test = train_test_split(df,test_size = 0.3,random_state = 10)


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


def train():
    privileged_groups = [{'race': 1}]
    unprivileged_groups = [{'race': 0}]
    optim_options = {
        "distortion_fun": get_distortion_syn,
        "epsilon": 0.04,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }
    
    metric_transf_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups = unprivileged_groups,
                      privileged_groups = privileged_groups)
    
    OP = OP.fit(dataset_orig_train)
    
    #for j in dataset_orig_train.features: 
    #    del j[dataset_orig_train.feature_names.index('priors_count=missing')] 
    #dataset_orig_train.feature_names.remove('priors_count=missing')
    #
    #for j in dataset_orig_vt.features: 
    #    del j[dataset_orig_vt.feature_names.remove('priors_count=missing')] 
    #dataset_orig_vt.feature_names.remove('priors_count=missing')
    
    dataset_transf_cat_test = OP.transform(dataset_orig_vt, transform_Y = True)
    dataset_transf_cat_test = dataset_orig_vt.align_datasets(dataset_transf_cat_test)
    
    
    dataset_transf_cat_train = OP.transform(dataset_orig_train, transform_Y = True)
    dataset_transf_cat_train = dataset_orig_train.align_datasets(dataset_transf_cat_train)
    
    
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_cat_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    print(metric_transf_train.mean_difference())
    
    
    
    
    
    scale_transf = StandardScaler()
    #X_train = scale_transf.fit_transform(dataset_transf_cat_train.features[:,1:])
    X_train = scale_transf.fit_transform(dataset_transf_cat_train.features)
    y_train = dataset_transf_cat_train.labels.ravel()
    
    #X_test = scale_transf.fit_transform(dataset_transf_cat_test.features[:,1:])
    X_test = scale_transf.fit_transform(dataset_transf_cat_test.features)
    
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train)
    y_pred = lmod.predict(X_test)
    print('Without reweight')
    print('Accuracy')
    print(accuracy_score(dataset_orig_vt.labels, y_pred))
    
    dataset_orig_vt_copy1 = dataset_orig_vt.copy()
    dataset_orig_vt_copy1.labels = y_pred
    
    metric_transf_train1 = BinaryLabelDatasetMetric(dataset_orig_vt_copy1, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    print('p-rule')
    print(metric_transf_train1.disparate_impact())
    print('CV')
    print(metric_transf_train1.mean_difference())
    print('FPR for unpriv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],0))
    print("FNR for unpriv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],0))
    
    print('FPR for priv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],0))
    print("FNR for priv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],0))
    df_weight = dataset_orig_train.convert_to_dataframe()[0]
    df_weight['weight'] = 1
    df_weight['is_missing'] = 0
    df_weight['tmp'] = ''
    tmp_result = []
    for i,j in zip(df_weight['race'],df_weight['two_year_recid']):
        tmp_result.append(str(i)+str(j))
    df_weight['tmp'] = tmp_result
    
    
    df_weight.loc[df_weight['priors_count=missing']==1,'is_missing'] = 1
    
    for i in df_weight['tmp'].unique():
        df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==0),'weight'] = len(df_weight.loc[(df_weight['tmp']==i),:].index)/len(df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==0),:].index)
        df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==1),'weight'] = len(df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==0),:].index)/len(df_weight.loc[(df_weight['tmp']==i),:].index)
    dataset_orig_train.instance_weights = np.array(df_weight['weight'])
    
    
    scale_transf = StandardScaler()
    #X_train = scale_transf.fit_transform(dataset_transf_cat_train.features[:,1:])
    X_train = scale_transf.fit_transform(dataset_transf_cat_train.features)
    y_train = dataset_transf_cat_train.labels.ravel()
    
    #X_test = scale_transf.fit_transform(dataset_transf_cat_test.features[:,1:])
    X_test = scale_transf.fit_transform(dataset_transf_cat_test.features)
    
    lmod = LogisticRegression()
    lmod.fit(X_train, y_train,sample_weight=dataset_orig_train.instance_weights)
    y_pred = lmod.predict(X_test)
    print('With reweight')
    print('Accuracy')
    print(accuracy_score(dataset_orig_vt.labels, y_pred))
    
    
    dataset_orig_vt_copy1 = dataset_orig_vt.copy()
    dataset_orig_vt_copy1.labels = y_pred
    
    metric_transf_train1 = BinaryLabelDatasetMetric(dataset_orig_vt_copy1, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    print('p-rule')
    print(metric_transf_train1.disparate_impact())
    print('CV')
    print(metric_transf_train1.mean_difference())
    print('FPR for unpriv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],0))
    print("FNR for unpriv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],0))
    print('FPR for priv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],0))
    print("FNR for priv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],0))

    
if __name__ == "__main__":  
    train()