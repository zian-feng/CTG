# packages & dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter
import seaborn as sb

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from skorch import NeuralNetClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import svm

from imblearn.over_sampling import ADASYN

from skorch.callbacks import EarlyStopping
from imblearn.metrics import classification_report_imbalanced
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


## Data Wrangling  
# **************************************************************************************


# load data
data = pd.read_excel('data/CTG.xls', engine = 'xlrd', sheet_name = 1, skiprows = 1)            # requires 'xlrd' package for import

# preview of head and tail
data.head(-1)

# remove empty footer rows
data = data.iloc[0:2126, :]

# remove empty columns 'unnamed'
data.columns                                                       # inspect col names
data = data.loc[:, ~data.columns.str.contains('^Unnamed:')]        # ~ bitwise negation operator

# removing non feature columns
data.columns.get_loc('LB')              # get index of first feature
data.columns.get_loc('Tendency')        # get index of last feature

# left join features and target variables
data = data.iloc[:,9:30].join(data.NSP, how = 'left')


# data normalization

data.head(-1)

'''
- after examining the data frame, the variables `ASTV` and `ALTV` are percentages so we will normalize them by 100
- LB, AC, FM, UC, DL, DS, and DP will be scaled using min-max (no negative values)
- the rest of the data will be standardised by z score
- the last feature column `tendency` will be left out because it is already scaled -1 to 1
'''

# re-scaled percentages features to decimal 
data.ASTV = data.ASTV/100
data.ALTV = data.ALTV/100

# apply min-max scaling to first seven features (baseline should not be below 0)
scaler = MinMaxScaler()
data[data.columns[0:7]] = scaler.fit_transform(data[data.columns[0:7]])


# standardise other features by z-score
columns = ['MSTV', 'MLTV', 'Width', 'Min', 'Max', 'Nmax',
           'Nzeros', 'Mode', 'Mean','Median', 'Variance']

for col in columns:
    scaler = StandardScaler()
    data[col] = scaler.fit_transform(data[[col]])

data.head(-1)


# Feature Correlation and Feature Selection
# **************************************************************************************
# 1. correlation-based feature selection
# 2. VIF based feature selection by recursive feature elimination


features = data.iloc[:,:-1]

corr = features.corr()          # pearson correlation

sb.set_theme(style="white")
f, ax = plt.subplots(figsize = (9, 9))
mask = np.triu(np.ones_like(corr, dtype = bool))
cmap = sb.diverging_palette(230, 20, as_cmap = True)
sb.heatmap(corr, mask = mask, cmap = cmap, vmin = -1, vmax = 1, center=0,
            square = True, linewidths = .5, annot = True, fmt = '0.2f', 
            cbar_kws = {"shrink": .5}, annot_kws = {'size': 5, 'rotation': 25})
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, ha = 'right')
plt.show()


# check for multi-collinearity
# remove highly correlated features
# use vif to recursively remove highest vif

def rfe_vif(data, threshold):
    features = data.iloc[:, :-1].copy()

    while True:
        vif = pd.DataFrame()                        # initialize empty df
        vif['var'] = features.columns
        vif['vif'] = [variance_inflation_factor(features.values, i) 
                  for i in range(features.shape[1])]

        vmax = vif['vif'].max()                     # find max vif        
        if vmax > threshold:
            var_drop = vif.sort_values('vif', ascending = False).iloc[0]['var']
            print(f'feature: {var_drop} dropped with VIF = {vmax}')
            features = features.drop(columns = [var_drop])
        
        else:
            break
    return features.columns

rfe_vif(features, threshold = 10)       # returns list of remaining features


# join target variable `data.NSP`
data = data.loc[:, ('AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1', 'ASTV', 'MSTV', 'ALTV',
       'MLTV', 'Min', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Variance', 'NSP')]


# encoding target variable
data.NSP.unique()
# encoding target variable y
# re-encoding normal as 0, suspected as 1, pathological as 2
data.NSP = data.NSP - 1

# preview target class proportions
sum(data.NSP==0)                                   # 1655
sum(data.NSP==1)                                   # 295
sum(data.NSP==2)                                   # 176

p0 = sum(data.NSP==0) / len(data.NSP)              # prop0 = 0.7784571966133584
p1 = sum(data.NSP==1) / len(data.NSP)              # prop1 = 0.138758231420508
p2 = sum(data.NSP==2) / len(data.NSP)              # prop2 = 0.08278457196613359

# plotting proportions
p_labels = ['Normal', 'Suspected', 'Pathological']
prop = [p0, p1, p2]

plt.figure(figsize = (5, 5))
plt.pie(prop, labels = p_labels, autopct = '%1.2f%%', 
        startangle = 45, colors = [cm.get_cmap('Pastel1')(i) for i in range(2,5)])
plt.title('Class Proportions')
plt.show()

# Data Partitioning
# **************************************************************************************

# Create Train & Final Test Set Split
ctg_train, ctg_test = train_test_split(data, test_size = 0.2, train_size = 0.8,
                                         random_state = 100, shuffle = True,  stratify = data.NSP)

# ctg_test will not be used until the final model evaluations
# we will save ctg_test as `test.csv` and remove it from memory
# we will re-import test.csv at the evaluation stage 

X_test = ctg_test.iloc[:, :-1]
y_test = ctg_test.iloc[:, -1]


# save to test set as csv
# ctg_test.to_csv('data/test.csv', index = False)

# Data Partitioning part II
# **************************************************************************************
'''
here we will use ctg_train to generate a class-balanced dateset
using adasyn to synthetically oversample the two minority classes

the majority is kept the same, the two minority classes are sampled

we will name the dataframe 'ctg_syn'

note this dataset is used to train and optimize weights
'''

# define features X and target y for training data
X = ctg_train.iloc[:, :-1]
y = ctg_train.iloc[:, -1]

# define X and y imbalanced variables
X_imb = ctg_train.iloc[:, :-1]
y_imb = ctg_train.iloc[:, -1]

# addressing the class imbalance issue:

# initiate sampler
ada = ADASYN(random_state = 100)                        # look at sampling strategy parameter
# sampling strategy params: str -- 'not majority' -> same as 'auto'
# this resamples the minority classes except the majority class
# n_neighbours -> evaluate this

X_syn, y_syn = ada.fit_resample(X, y)

X = X_syn               # define X, y to be class-balanced training data
y = y_syn

# define k fold cv
k = 6
kf = KFold(n_splits = k, shuffle = True, random_state = 100)

# SVM MODEL
# **************************************************************************************

# base model before hyper parameter tuning
# use imbalanced data for base models

SVM_base = svm.SVC(kernel='linear', random_state = 100)
SVM_base.fit(X_imb, y_imb)                                   # fit on imbalanced data
SVM_base.get_params()
pred_svm_b = SVM_base.predict(X_test)

accuracy_score(y_test, pred_svm_b)
balanced_accuracy_score(y_test, pred_svm_b)
precision_score(y_test, pred_svm_b, average='macro')
recall_score(y_test, pred_svm_b, average='macro')
f1_score(y_test, pred_svm_b, average='macro')

classification_report(y_test, pred_svm_b)
classification_report_imbalanced(y_test, pred_svm_b)

''' Results: classification report
              precision    recall  f1-score   support

         0.0       0.93      0.95      0.94       332
         1.0       0.65      0.61      0.63        59
         2.0       0.79      0.66      0.72        35

    accuracy                           0.88       426
   macro avg       0.79      0.74      0.76       426
weighted avg       0.88      0.88      0.88       426
'''
''' Results: classification report imbalanced
                   pre       rec       spe        f1       geo       iba       sup

        0.0       0.93      0.95      0.73      0.94      0.84      0.72       332
        1.0       0.65      0.61      0.95      0.63      0.76      0.56        59
        2.0       0.79      0.66      0.98      0.72      0.80      0.63        35

avg / total       0.88      0.88      0.78      0.88      0.82      0.69       426
'''

# SVM architecture hyper-tuning
# initialize SVM Classifier
classifier = svm.SVC(random_state = 100)


# tune architectural hyperparameters with gridsearch
classifier.get_params() # view tunable parameters

hypergrid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'degree': [2, 3, 4, 5],                                 # only applicable to poly kernel
    'decision_function_shape': ['ovo', 'ovr'],              # decision function using 'ovr' for multiclass
    }

gs_svm_1 = GridSearchCV(estimator = classifier,
                        param_grid = hypergrid,
                        cv = kf,
                        verbose = True,
                        scoring = 'f1_weighted'
                        )

gs_svm_1.fit(X, y)           

gs_svm_1.best_params_           # {'decision_function_shape': 'ovr','degree': 2, 'gamma': 'scale', 'kernel': 'rbf'}
gs_svm_1.best_score_            # f1_weighted: 0.8972

gs_svm_1.cv_results_.keys()
gs_svm_1.cv_results_['mean_test_score']
gs_svm_1.cv_results_['params']


# define SVM Classifier w/ hyperparameter architecture
# note degree = 2 can be omitted because it is ignored by rbf kernel

SVM_h1 = svm.SVC(kernel = 'rbf', gamma = 'scale',
                   decision_function_shape = 'ovr',
                   random_state = 100)


SVM_h1.fit(X, y)

pred_svm_h1 = SVM_h1.predict(X_test)

accuracy_score(y_test, pred_svm_h1)
balanced_accuracy_score(y_test, pred_svm_h1)
precision_score(y_test, pred_svm_h1, average='macro')
recall_score(y_test, pred_svm_h1, average='macro')
f1_score(y_test, pred_svm_h1, average='macro')

classification_report(y_test, pred_svm_h1)
classification_report_imbalanced(y_test, pred_svm_h1)

# intermediate results
''' Results: classification report
              precision    recall  f1-score   support

         0.0       0.96      0.91      0.94       332
         1.0       0.59      0.75      0.66        59
         2.0       0.81      0.86      0.83        35

    accuracy                           0.88       426
   macro avg       0.79      0.84      0.81       426
weighted avg       0.90      0.88      0.89       426
'''
''' Results: classification report imbalanced
                   pre       rec       spe        f1       geo       iba       sup

        0.0       0.96      0.91      0.88      0.94      0.90      0.81       332
        1.0       0.59      0.75      0.92      0.66      0.83      0.67        59
        2.0       0.81      0.86      0.98      0.83      0.92      0.83        35

avg / total       0.90      0.88      0.90      0.89      0.89      0.79       426
'''

# grid search for optimization hyperparameters
classifier = svm.SVC(kernel = 'rbf', gamma = 'scale', 
                     decision_function_shape = 'ovr',
                     random_state = 100)

hypergrid = {
    'C': [0.01, 0.1, 1, 10, 100],             # regularization
    'tol': [1e-3, 1e-4, 1e-5],                # tolerance
    'max_iter': [10000],                      # max iterations
    #'class_weight': ['balanced']              # penalty to specific classes
    }

gs_svm_2 = GridSearchCV(estimator = classifier,
                        param_grid = hypergrid,
                        cv = kf,
                        verbose = True,
                        scoring='f1_weighted'
                        )


gs_svm_2.fit(X, y)

gs_svm_2.best_params_             # {'C': 10, 'class_weight': 'balanced', 'max_iter': 10000, 'tol': 0.001}
gs_svm_2.best_score_              # 0.9610

gs_svm_2.cv_results_              # cross validation results


# FINAL TUNED SVM MODEL
SVM_final = svm.SVC(kernel = 'rbf', gamma = 'scale',
                   decision_function_shape = 'ovr',
                   C = 100, class_weight = 'balanced',
                   max_iter = 10000,
                   tol = 0.001,
                   random_state = 100)

# training
SVM_final.fit(X, y)

y_pred = SVM_final.predict(X_test)
y_pred_svm = y_pred.copy()                                  # save predictions

balanced_accuracy_score(y_test, y_pred_svm)
accuracy_score(y_test, y_pred_svm)
precision_score(y_test, y_pred_svm, average='macro')
recall_score(y_test, y_pred_svm, average='macro')
f1_score(y_test, y_pred_svm, average='macro')

classification_report(y_test, y_pred_svm)
classification_report_imbalanced(y_test, y_pred_svm)

''' Results: classification report
                precision    recall  f1-score   support

         0.0       0.96      0.96      0.96       332
         1.0       0.72      0.69      0.71        59
         2.0       0.86      0.86      0.86        35

    accuracy                           0.92       426
   macro avg       0.84      0.84      0.84       426
weighted avg       0.91      0.92      0.91       426
'''
''' Results: classification report imbalanced
                   pre       rec       spe        f1       geo       iba       sup

        0.0       0.96      0.96      0.84      0.96      0.90      0.82       332
        1.0       0.72      0.69      0.96      0.71      0.82      0.65        59
        2.0       0.86      0.86      0.99      0.86      0.92      0.84        35

avg / total       0.91      0.92      0.87      0.91      0.89      0.80       426
'''

# ANN MODEL
# **************************************************************************************

# convert to tensor
# use imbalanced values
X = ctg_train.iloc[:, :-1]
y = ctg_train.iloc[:, -1]

X_tensor = torch.tensor(X.values, dtype = torch.float32)
y_tensor = torch.tensor(y.values, dtype = torch.long)

X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32)


# base model
class MLP_base(nn.Module):
    def __init__(self, h_dim):
        super(MLP_base, self).__init__()

        self.input = nn.Linear(16, h_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(h_dim, 3)
        
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.out(x)

        return x
    

# we omit the softmax layer explicitly because criterion nn.CrossEntropyLoss applies softmax

MLP = NeuralNetClassifier(
    MLP_base,
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.77, 0.13, 0.10], dtype = torch.float32)),
    max_epochs = 20,
    module__h_dim = 16
)

MLP.fit(X_tensor, y_tensor)

pred_ann_b = MLP.predict(X_test_tensor)

accuracy_score(y_test, pred_ann_b)
balanced_accuracy_score(y_test, pred_ann_b)
precision_score(y_test, pred_ann_b, average='weighted')
recall_score(y_test, pred_ann_b, average='macro')
f1_score(y_test, pred_ann_b, average='macro')

classification_report(y_test, pred_ann_b)
classification_report_imbalanced(y_test, pred_ann_b)

''' Results: classification report
              precision    recall  f1-score   support

         0.0       0.78      1.00      0.88       332
         1.0       0.00      0.00      0.00        59
         2.0       0.00      0.00      0.00        35

    accuracy                           0.78       426
   macro avg       0.26      0.33      0.29       426
weighted avg       0.61      0.78      0.68       426
'''
''' Results: classification report imbalanced
                   pre       rec       spe        f1       geo       iba       sup

        0.0       0.78      1.00      0.00      0.88      0.00      0.00       332
        1.0       0.00      0.00      1.00      0.00      0.00      0.00        59
        2.0       0.00      0.00      1.00      0.00      0.00      0.00        35

avg / total       0.61      0.78      0.22      0.68      0.00      0.00       426
'''

# architectural hyper-tuning 

# convert data to torch tensors for neural network
X = X_syn
y = y_syn

X_tensor = torch.tensor(X.values, dtype = torch.float32)
y_tensor = torch.tensor(y.values, dtype = torch.long)

# set seed for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class ANN_base(nn.Module):
    def __init__(self, h_dim, n_layers):
        super(ANN_base, self).__init__()

        self.input = nn.Linear(16, h_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(h_dim, 3)

        self.h_layers = nn.ModuleList([
            nn.Linear(h_dim, h_dim) for _ in range(n_layers)])
        
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)

        for layer in self.h_layers:
            x = layer(x)
            x = self.relu(x)
        
        x = self.out(x)

        return x


basenet = NeuralNetClassifier(
    ANN_base,
    criterion = nn.CrossEntropyLoss,
    max_epochs = 30,
    callbacks = [EarlyStopping(monitor = 'valid_loss', patience = 5, 
                      threshold = 0.0001, lower_is_better = True)]
)

arc_grid = {
    'module__n_layers': [1, 2, 3],
    'module__h_dim': [4, 6, 8, 10, 12, 14, 16],
}

arc_gs = GridSearchCV(basenet, arc_grid, cv = kf, scoring = 'f1_weighted')

arc_gs.fit(X_tensor, y_tensor)
arc_gs.best_params_
arc_gs.best_score_      

# best n_layers: 1 hidden layer
# best n_neurons: 10

# redefine model for optimizatin hypertuning
class ANN_base_model(nn.Module):
    def __init__(self):
        super(ANN_base_model, self).__init__()

        self.fc1 = nn.Linear(16, 10)            # input layer
        self.relu = nn.ReLU()
        self.h1 = nn.Linear(10, 10)         # hidden layer 1
        self.fc2 = nn.Linear(10, 3)                 # output layer
        
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.h1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

ANN_b = NeuralNetClassifier(
    ANN_base_model,
    criterion = nn.CrossEntropyLoss,
    max_epochs = 50,
    callbacks = [EarlyStopping(monitor = 'valid_loss', patience = 5, 
                      threshold = 0.0001, lower_is_better = True)]
    )

X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32)

ANN_b.fit(X_tensor, y_tensor)

y_pred_h1 = ANN_b.predict(X_test_tensor)

accuracy_score(y_test, y_pred_h1)
balanced_accuracy_score(y_test, y_pred_h1)
precision_score(y_test, y_pred_h1, average='weighted')
recall_score(y_test, y_pred_h1, average='macro')
f1_score(y_test, y_pred_h1, average='macro')

classification_report(y_test, y_pred_h1)
balanced_accuracy_score(y_test, y_pred_h1)

# intermediate results
''' Results: classification report
              precision    recall  f1-score   support

         0.0       0.91      0.33      0.49       332
         1.0       0.00      0.00      0.00        59
         2.0       0.11      0.97      0.20        35

    accuracy                           0.34       426
   macro avg       0.34      0.43      0.23       426
weighted avg       0.72      0.34      0.39       426
'''
''' Results: classification report imbalanced
                  pre       rec       spe        f1       geo       iba       sup

        0.0       0.91      0.33      0.88      0.49      0.54      0.28       332
        1.0       0.00      0.00      1.00      0.00      0.00      0.00        59
        2.0       0.11      0.97      0.31      0.20      0.55      0.32        35

avg / total       0.72      0.34      0.85      0.39      0.47      0.24       426
'''

##### gridsearch ANN FINAL MODEL

# for optimization hyper parameters

class ANN_model(nn.Module):
    def __init__(self, dr):
        super(ANN_model, self).__init__()

        self.input = nn.Linear(16, 12)
        self.relu = nn.ReLU()
        self.h1 = nn.Linear(12, 12)
        self.d = nn.Dropout(dr)
        self.output = nn.Linear(12,3)

    def forward(self, x):

        x = self.input(x)
        x = self.relu(x)
        x = self.h1(x)
        x = self.relu(x)
        x = self.d(x)
        x = self.output(x)
        return x
    
ANNnetwork = NeuralNetClassifier(
    ANN_model,
    device = 'cpu',
    criterion= nn.CrossEntropyLoss,
    max_epochs = 50,
    callbacks = [EarlyStopping(monitor = 'valid_loss', patience = 5, 
                      threshold = 0.0001, lower_is_better = True)],
)


hypergrid = {
    'lr': [0.1, 0.01, 0.001],
    'module__dr': [0, 0.2, 0.4, 0.5, 0.8],
    'optimizer': [optim.SGD, optim.Adam, optim.Rprop],
    'optimizer__weight_decay': [0, 1e-2, 1e-3, 1e-4],
    'optimizer__momentum': [0, 0.25, 0.5, 0.75, 0.85, 0.95],
    'batch_size': [32, 64, 96, 128]
}

gs = GridSearchCV(ANNnetwork, hypergrid, cv = kf, scoring = 'f1_weighted')
# took over 2 hours to run
# running gridsearch using weighted f1 scoring metric as our final holdout is imbalanced

gs.fit(X_tensor, y_tensor)

gs.best_params_
gs.best_score_

''' Gridsearch Results:
'lr': 0.01, 
'module__dr': 0.2, 
'optimizer': <class 'torch.optim.sgd.SGD'>, 
'optimizer__momentum': 0.95, 
'optimizer__weight_decay': 0
'batch_size': 128 
'''

# define final ANN model

# implement LR scheduler for plateau

ANN_final = NeuralNetClassifier(
    ANN_model,
    lr = 0.01,
    module__dr = 0.2,
    criterion = nn.CrossEntropyLoss,
    optimizer = optim.SGD,
    optimizer__momentum = 0.95,
    optimizer__weight_decay = 0,
    max_epochs = 50,
    batch_size = 128,
    callbacks = [EarlyStopping(monitor = 'valid_loss', patience = 5, 
                      threshold = 0.0001, lower_is_better = True),
                      LRScheduler(policy = ReduceLROnPlateau, 
                                  monitor = 'valid_loss', factor = 0.5, patience = 3)]
)

ANN_final.fit(X_tensor, y_tensor)

y_pred_ann = ANN_final.predict(X_test_tensor)

accuracy_score(y_test, y_pred_ann)
balanced_accuracy_score(y_test, y_pred_ann)                       # 0.8749127259643902  
precision_score(y_test, y_pred_ann, average='weighted')
recall_score(y_test, y_pred_ann, average='weighted')
f1_score(y_test, y_pred_ann, average='weighted')

classification_report(y_test, y_pred_ann)
classification_report_imbalanced(y_test, y_pred_ann)

''' results: classification report
              precision    recall  f1-score   support

         0.0       0.99      0.89      0.94       332
         1.0       0.59      0.85      0.69        59
         2.0       0.76      0.89      0.82        35

    accuracy                           0.88       426
   macro avg       0.78      0.87      0.82       426
weighted avg       0.91      0.88      0.89       426
'''
''' results: classification report imbalanced
                   pre       rec       spe        f1       geo       iba       sup

        0.0       0.99      0.89      0.96      0.94      0.92      0.85       332
        1.0       0.59      0.85      0.90      0.69      0.88      0.76        59
        2.0       0.76      0.89      0.97      0.82      0.93      0.86        35

avg / total       0.91      0.88      0.95      0.89      0.92      0.84       426
'''

## visualizations
# **************************************************************************************

# ann base model loss

MLP.history
t_loss = [batch['train_loss'] for batch in MLP.history]
v_loss = [batch['valid_loss'] for batch in MLP.history]

epoch_range = range(1, len(t_loss) + 1)

plt.figure(figsize = (12, 5))
plt.plot(epoch_range, t_loss, 'x-', label = 'Training Loss')
plt.plot(epoch_range, v_loss, 'x-', label = 'Validation Loss')
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.title('Training & Validation Loss per Epoch Base Model (ANN)')
plt.legend()
plt.grid(True)
plt.show()


# final ann loss

ANN_final.history
t_loss = [batch['train_loss'] for batch in ANN_final.history]
v_loss = [batch['valid_loss'] for batch in ANN_final.history]

epoch_range = range(1, len(t_loss) + 1)

plt.figure(figsize = (12, 5))
plt.plot(epoch_range, t_loss, 'x-', label = 'Training Loss')
plt.plot(epoch_range, v_loss, 'x-', label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss per Batch Final Model (ANN)')
plt.legend()
plt.grid(True)
plt.show()

# classification confusion matrix for 3 classes

# ann
cm = confusion_matrix(y_true = y_test, y_pred = y_pred_ann )
cm_ann = pd.DataFrame(cm, 
                      index = ['normal', 'suspected', 'pathological)'],
                      columns = ['normal', 'suspected', 'pathological'])

plt.figure(figsize=(5,4))
ax = sb.heatmap(cm_ann, annot=True, fmt = 'g', cmap = 'Blues', center = None, cbar = False)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.title('Confusion Matrix for ANN Final Model')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.tight_layout()
# plt.savefig('cm_ann.png', dpi = 300)
plt.show()

# svm
cm = confusion_matrix(y_true = y_test, y_pred = y_pred_svm )
cm_svm = pd.DataFrame(cm, 
                      index = ['normal', 'suspected', 'pathological'],
                      columns = ['normal', 'suspected', 'pathological'])

plt.figure(figsize=(5,4))
ax = sb.heatmap(cm_svm, annot=True, fmt = 'g', cmap = 'Blues', center = None, cbar = False)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
plt.title('Confusion Matrix for SVM Final Model')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.tight_layout()
# plt.savefig('cm_svm.png', dpi = 300)
plt.show()


# ROC - AUC curves

y_test_bin = label_binarize(y_test, classes = y_test.unique())

# get probabilistic predictions for each model
y_score_ANN = ANN_final.predict_proba(X_test_tensor)
y_score_SVM = SVM_final.decision_function(X_test)

# create empty dictionaries
fpr_ann, tpr_ann, auc_ann = {}, {}, {}
fpr_svm, tpr_svm, auc_svm = {}, {}, {}

# compute ROC and AUC by class
for i in range(3):
    fpr_ann[i], tpr_ann[i], _ = roc_curve(y_test_bin[:, i], y_score_ANN[:, i])
    auc_ann[i] = auc(fpr_ann[i], tpr_ann[i])

    fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test_bin[:, i], y_score_SVM[:, i])
    auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])

# macro & micro averaged ROC-AUC scores
auc_ann_ovr_macro = roc_auc_score(y_test_bin, y_score_ANN, multi_class = 'ovr', average = 'macro')
auc_svm_ovr_macro = roc_auc_score(y_test_bin, y_score_SVM, multi_class = 'ovr', average = 'macro')

auc_ann_ovr_micro = roc_auc_score(y_test_bin, y_score_ANN, multi_class = 'ovr', average = 'micro')
auc_svm_ovr_micro = roc_auc_score(y_test_bin, y_score_SVM, multi_class = 'ovr', average = 'micro')

auc_ann
auc_svm
auc_ann_ovr_macro
auc_svm_ovr_macro 
auc_ann_ovr_micro
auc_svm_ovr_micro

# plot ROC-AUC

class_labels = ['class N = 0', 'class S = 1', 'class P = 2']

plt.figure(figsize = (10, 6))
for i in range(3):
    plt.plot(fpr_ann[i], tpr_ann[i], linestyle = '-', label = f'ANN {class_labels[i]} (AUC: {auc_ann[i]: .4f})')
    plt.plot(fpr_svm[i], tpr_svm[i], linestyle = '-', label = f'SVM {class_labels[i]} (AUC: {auc_svm[i]: .4f})')

plt.plot([0, 1], [0, 1], color = 'gray', linestyle = ':', label = 'Random (AUC = 0.50)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc = 'lower right')
plt.tight_layout()
# plt.savefig('ROC.png', dpi = 300)
plt.show()


# exporting models and saving parameters
# use .pkl

# import joblib
# joblib.dump(SVM_final, 'svm.pkl')

# torch.save(ANN_final.module_.state_dict(), 'ann_weights.pt')
# ANN_final.save_params(f_params = 'ann_params.pkl')



# code references
''' 
[three class cm](https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/)
[roc curves]((https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#roc-curve-using-the-ovr-macro-average))
[NeuralNetClassifier](https://skorch.readthedocs.io/en/latest/classifier.html)
[skorch callbacks](https://skorch.readthedocs.io/en/stable/callbacks.html)
[ADASYN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)
[variance inflation factor](https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)
'''