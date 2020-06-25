import sys
sys.path.append("../")
sys.path.append("AIF360/")
import numpy as np
from tqdm import tqdm
import pandas as pd 
from fairness_metrics.tot_metrics import TPR,TNR,get_BCR,DIbinary
import random
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from common_utils import compute_metrics

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from aif360.datasets import StandardDataset
import warnings
warnings.simplefilter("ignore")

def get_distortion_adult(vold, vnew):
    
    """Distortion function for the adult dataset. We set the distortion
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

    # Define local functions to adjust education and age
    def adjustEdu(v):
        if v == '>12':
            return 13
        elif v == '<6':
            return 5
        elif v =='missing_edu':
            return -1
        else:
            return int(v)

    def adjustAge(a):
        if a == '>=70':
            return 70.0
        else:
            return float(a)

    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)

    # value that will be returned for events that should not occur
    bad_val = 3.0

    # Adjust education years
    eOld = adjustEdu(vold['Education Years'])
    eNew = adjustEdu(vnew['Education Years'])

    # Education cannot be lowered or increased in more than 1 year
    if eNew ==-1:
        if eOld ==-1:
            return 1.0
        else: 
            return bad_val
    elif eOld ==-1:
        if eNew ==-1:
            return 1.0
        else:
            return 0.0
    elif (eNew < eOld) | (eNew > eOld+1):
        return bad_val

    # adjust age
    aOld = adjustAge(vold['Age (decade)'])
    aNew = adjustAge(vnew['Age (decade)'])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld-aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld-aNew) > 5:
        return 2.0

    # Adjust income
    incOld = adjustInc(vold['Income Binary'])
    incNew = adjustInc(vnew['Income Binary'])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0


default_mappings = {
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'}]
}

class AdultDataset(StandardDataset):
    """Adult Census Income Dataset.

    See :file:`aif360/data/raw/adult/README.md`.
    """

    def __init__(self, label_name='income-per-year',
                 favorable_classes=['>50K', '>50K.'],
                 protected_attribute_names=['sex'],
                 privileged_classes=[['Male']],
                 instance_weights_name=None,
                 categorical_features=['workclass', 'education',
                     'marital-status', 'occupation', 'relationship',
                     'native-country'],
                 features_to_keep=[], features_to_drop=['fnlwgt'],
                 na_values=['?'], custom_preprocessing=None,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.

        Examples:
            The following will instantiate a dataset which uses the `fnlwgt`
            feature:

            >>> from aif360.datasets import AdultDataset
            >>> ad = AdultDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            >>> not np.all(ad.instance_weights == 1.)
            True

            To instantiate a dataset which utilizes only numerical features and
            a single protected attribute, run:

            >>> single_protected = ['sex']
            >>> single_privileged = [['Male']]
            >>> ad = AdultDataset(protected_attribute_names=single_protected,
            ... privileged_classes=single_privileged,
            ... categorical_features=[],
            ... features_to_keep=['age', 'education-num'])
            >>> print(ad.feature_names)
            ['education-num', 'age', 'sex']
            >>> print(ad.label_names)
            ['income-per-year']

            Note: the `protected_attribute_names` and `label_name` are kept even
            if they are not explicitly given in `features_to_keep`.

            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> ad = AdultDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        train_path = 'AIF360/aif360/data/raw/adult/adult.data'
        test_path = 'AIF360/aif360/data/raw/adult/adult.test'
                # as given by adult.names
        column_names = ['age', 'workclass', 'fnlwgt', 'education',
            'education-num', 'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country', 'income-per-year']
        
        train = pd.read_csv(train_path, header=None, names=column_names,
            skipinitialspace=True, na_values=na_values)
        test = pd.read_csv(test_path, header=0, names=column_names,
            skipinitialspace=True, na_values=na_values)

        df = pd.concat([train, test], ignore_index=True)
        
        

        super(AdultDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)




def load_preproc_data_adult(protected_attributes=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
        """
        np.random.seed(1)
        # Group age by decade
        df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def group_edu(x):
            if x == -1:
                return 'missing_edu'
            elif x <= 5:
                return '<6'
            elif x >= 13:
                return '>12'
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
        df['Education Years'] = df['Education Years'].astype('category')

        # Limit age range
        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        # Rename income variable
        df['Income Binary'] = df['income-per-year']

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        df1 = df[['sex','Education Years','Age (decade)','Income Binary']]
        tot = []
        for index, row in df1.iterrows():
            result = ''
            for j in df1.columns:
                result = result+str(row[j])
            tot.append(result)
        df1['tmp_feature'] = tot
        df1['mis_prob'] = 0
        for i in df1['tmp_feature'].unique():
            if '<=50K' in i and i[0]=='0':
                df1.loc[df1['tmp_feature']==i,'mis_prob'] = 0.8
            elif i[0]=='1':
                df1.loc[df1['tmp_feature']==i,'mis_prob'] = 0.08
            else:
                df1.loc[df1['tmp_feature']==i,'mis_prob'] = 0.04
        new_label=[]
        for i,j in zip(df1['mis_prob'],df1['Education Years']):
            if np.random.binomial(1, i, 1)[0]==1:
                new_label.append(-1)
            else:
                new_label.append(j)
        df['Education Years'] = new_label
        print('Total number of missing values')
        print(len(df.loc[df['Education Years']==-1,:].index))
        print('Total number of observations')
        print(len(df.index))
        return df

    XD_features = ['Age (decade)', 'Education Years', 'sex']
    D_features = ['sex'] if protected_attributes is None else protected_attributes
    Y_features = ['Income Binary']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['Age (decade)', 'Education Years']

    # privileged classes
    all_privileged_classes = {"sex": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'}}

    return AdultDataset(
        label_name=Y_features[0],
        favorable_classes=['>50K', '>50K.'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def train():
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig = load_preproc_data_adult(['sex'])
       
    optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.02,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
    }

    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)


    OP = OptimPreproc(OptTools, optim_options,
                      unprivileged_groups = unprivileged_groups,
                      privileged_groups = privileged_groups)

    OP = OP.fit(dataset_orig_train)
    dataset_transf_cat_test = OP.transform(dataset_orig_vt, transform_Y = True)
    dataset_transf_cat_test = dataset_orig_vt.align_datasets(dataset_transf_cat_test)


    dataset_transf_cat_train = OP.transform(dataset_orig_train, transform_Y = True)
    dataset_transf_cat_train = dataset_orig_train.align_datasets(dataset_transf_cat_train)


    scale_transf = StandardScaler()
    X_train = scale_transf.fit_transform(dataset_transf_cat_train.features)
    y_train = dataset_transf_cat_train.labels.ravel()
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
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],1))
    print("FNR for unpriv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],1))
    print('FPR for priv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],1))
    print("FNR for priv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],1))

    df_weight = dataset_orig_train.convert_to_dataframe()[0]
    df_weight['weight'] = 1
    df_weight['is_missing'] = 0
    df_weight['tmp'] = ''
    tmp_result = []
    for i,j in zip(df_weight['sex'],df_weight['Income Binary']):
        tmp_result.append(str(i)+str(j))
    df_weight['tmp'] = tmp_result


    df_weight.loc[df_weight['Education Years=-1']==1,'is_missing'] = 1

    for i in df_weight['tmp'].unique():
        df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==0),'weight'] = len(df_weight.loc[(df_weight['tmp']==i),:].index)/len(df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==0),:].index)
        df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==1),'weight'] = len(df_weight.loc[(df_weight['tmp']==i)&(df_weight['is_missing']==0),:].index)/len(df_weight.loc[(df_weight['tmp']==i),:].index)
    dataset_orig_train.instance_weights = np.array(df_weight['weight'])


    scale_transf = StandardScaler()
    X_train = scale_transf.fit_transform(dataset_transf_cat_train.features)
    y_train = dataset_transf_cat_train.labels.ravel()

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
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],1))
    print("FNR for unpriv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==0],y_pred[orig_sens_att==0],1))
    print('FPR for priv')
    orig_sens_att = dataset_orig_vt.protected_attributes.ravel()
    print(1-TNR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],1))
    print("FNR for priv")
    print(1-TPR(dataset_orig_vt.labels.ravel()[orig_sens_att==1],y_pred[orig_sens_att==1],1))

if __name__ == '__main__':
    train()