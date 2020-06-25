# Data and code repo for paper Quantifying the Impact of Data Bias in Machine Learning

The framework of our code uses code from another Github project at https://github.com/IBM/AIF360 with some modifications. 

This repository contains the following files: <br>
MAR_compas.py: accuracy and fairness measure before and after reweighting with MAR missing values using COMPAS data <br>
MAR_adult.py: accuracy and fairness measure before and after reweighting with MAR missing values using Adult data <br>
MAR_syn.py: accuracy and fairness measure before and after reweighting with MAR missing values using synthetic data <br>
MNAR_compas.py: accuracy and fairness measure before and after reweighting with MNAR missing values using COMPAS data <br>
MNAR_adult.py: accuracy and fairness measure before and after reweighting with MNAR missing values using Adult data <br>
MNAR_syn.py: accuracy and fairness measure before and after reweighting with MNAR missing values using synthetic data <br>
selection_compas_before_fix.py: accuracy and fairness measure before resampling with selection bias using COMPAS data <br>
selection_compas_after_fix.py: accuracy and fairness measure after resampling with selection bias using COMPAS data <br>
selection_adult_before_fix.py: accuracy and fairness measure before resampling with selection bias using Adult data <br>
selection_adult_after_fix.py: accuracy and fairness measure after resampling with selection bias using Adult data <br>
selection_syn_before_fix.py: accuracy and fairness measure before resampling with selection bias using synthetic data <br>
selection_syn_after_fix.py: accuracy and fairness measure after resampling with selection bias using synthetic data <br>
comb_compas_before_fix.py: accuracy and fairness measure before using fixing algorithms with both selection bias and missing values using COMPAS data <br>
comb_compas_stratified_resample.py: accuracy and fairness measure after using stratified resampling and reweighting with both selection bias and missing values (MAR) using COMPAS data <br>
comb_compas_unif_resample.py: accuracy and fairness measure after using uniform resampling and reweighting with both selection bias and missing values (MNAR) using COMPAS data <br>



# Reference
Rachel K. E. Bellamy Kuntal Dey and Michael Hind and Samuel C. Hoffman and Stephanie Houde and Kalapriya Kannan and Pranay Lohia and Jacquelyn Martino and Sameep Mehta and Aleksandra Mojsilovic and Seema Nagar and Karthikeyan Natesan Ramamurthy and John Richards and Diptikalyan Saha and Prasanna Sattigeri and Moninder Singh and Kush R. Varshney and Yunfeng Zhang: AI Fairness 360:  An Extensible Toolkit for Detecting, Understanding, and Mitigating Unwanted Algorithmic Bias (2018) https://arxiv.org/abs/1810.01943
