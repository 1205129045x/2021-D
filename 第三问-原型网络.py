import torch
import torch.nn as nn
import random
import torchvision
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib

torch.manual_seed(999)
np.random.seed(999)
random.seed(999)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='finish_model.pkl', trace_func=print):
        """
         Args:
         patience (int): How long to wait after last time validation loss improved.
         Default: 7
         verbose (bool): If True, prints a message for each validation loss improvement. 
         Default: False
         delta (float): Minimum change in the monitored quantity to qualify as an improvement.
         Default: 0
         path (str): Path for the checkpoint to be saved to.
         Default: 'checkpoint.pt'
         trace_func (function): trace print function.
         Default: print 
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:6f}). Saving model ...')
     # torch.save(model.state_dict(), self.path)
        torch.save(model, self.path)
        self.val_loss_min = val_loss

class MulitiPrototypicalNet3(nn.Module):
    def __init__(self,in_feature,num_class, embedding_dim,support_ratio = 0.6,query_ratio = 0.3,hidden1_dim = 1024,hidden2_dim = 256,distance='euclidean'):
        super(MulitiPrototypicalNet3, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.support_ratio = support_ratio
        self.query_ratio = query_ratio
        self.support_num = []
        self.query_num = []
        self.distance = distance
        self.prototype = None
        self.prototypes = []
        self.feature_extraction = nn.Sequential(
            nn.Linear(in_features=in_feature, out_features=hidden1_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden1_dim, out_features=hidden2_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden2_dim, out_features=embedding_dim),)
    def weights_init_(self):
        for m in self.modules():
            torch.nn.init.xavier_normal_(m.weight, gain=1, )
            torch.nn.init.constant_(m.bias, 0)  
    def embedding(self, features):
            result = self.feature_extraction(features)
            return result
    def forward(self, support_input, query_input):
            return result
    def randomGenerate(self, X, Y):
            return support_input, query_input, support_label, query_label
    def fit(self,X_train,y_train,X_valid,y_valid,optimizer,criterion,patience,EPOCH):
            return loss_list
        
    def predict(self,X_test):
            return pre_Y, prob_Y

def miv(model, X):
    model.eval()
    miv = torch.ones(X.shape[1])
    for i in range(X.shape[1]):
        cur_X_1 = X.copy()
        cur_X_2 = X.copy()
        cur_X_1[:, i] = cur_X_1[:, i] + cur_X_1[:, i] * 0.1
        cur_X_2[:, i] = cur_X_2[:, i] - cur_X_2[:, i] * 0.1
        cur_X_1 = torch.tensor(cur_X_1, dtype=torch.float)
        cur_X_2 = torch.tensor(cur_X_2, dtype=torch.float)
        cur_diff = torch.mean(model.embedding(cur_X_1) - model.embedding(cur_X_2), dim=1)
        miv[i] = torch.mean(cur_diff, dim=0)
    s = torch.abs(miv) / torch.sum(torch.abs(miv))
    rank = torch.argsort(torch.abs(miv), dim=0, descending=True)
    return rank, s

molecular_des = pd.read_csv("Molecular_Descriptor.csv",index_col=0)
admet = pd.read_csv("admet.csv",index_col=0)

X = molecular_des.values
y = admet.values

X_train, X_valid, y_train, y_valid = train_test_split( X, y, test_size=0.3, stratify=y,random_state=56)
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)
X_valid = ss.transform(X_valid)

joblib.dump(ss, 'p1_ss.pkl')
model = MulitiPrototypicalNet3(X_train.shape[1],2, 24,)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()
model.fit(X_train,y_train,X_valid,y_valid,optimizer,criterion,50,500)
pre_Y, prob_Y = model.predict(X_valid)
final_model1 = torch.load("./finish_model_1.pkl")
pre_valid_y_model1, prob_valid_y_model1 = final_model1.predict(X_valid)
acc = accuracy_score(y_valid, pre_valid_y_model1)
auc = roc_auc_score(y_valid,prob_valid_y_model1)
print('acc: ' + str(acc))
print('auc: ' + str(auc))