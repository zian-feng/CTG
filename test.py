# # import data
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau


# data = pd.read_excel('CTG.xls', engine = 'xlrd', sheet_name = 1, skiprows = 1)            # requires 'xlrd' package for import
data = pd.read_csv('data/test.csv')

# create X and Y
data.head(-1)

# convert to torch tensors
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_tensor = torch.tensor(X.values, dtype = torch.float32)
y_tensor = torch.tensor(Y.values, dtype = torch.long)



# import svm models
svm_model = joblib.load('models/svm.pkl')

# predict 
y_pred_svm = svm_model.predict(X)

print('SVM Model Predictions:', y_pred_svm)



# define ANN model
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


ann_model = NeuralNetClassifier(ANN_model, module__dr = 0.2)

# initialize model
ann_model.initialize()

# load parameters
ann_model.load_params(f_params = 'models/ann_params.pkl')

# load model weights
ann_model.module_.load_state_dict(torch.load('models/ann_weights.pt'))


# make predictions
y_pred_ann = ann_model.predict(X_tensor)

print('ANN Model Predictions:', y_pred_ann)

