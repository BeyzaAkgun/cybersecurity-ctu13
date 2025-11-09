import numpy as np  # for array
import pandas as pd  # for csv files and dataframe
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # plotting


# from category_encoders import hashing as hs # type: ignore
import itertools

import pickle  # To load data int disk
# from prettytable import PrettyTable  # type: ignore # To print in tabular format
import os

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# import xgboost as xgb

from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, classification_report, RocCurveDisplay 
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import auc, f1_score, roc_curve, roc_auc_score, recall_score, precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, cross_val_predict

# Load the dataset
binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
df = pd.read_csv(binetflow_file)
df = df.sample(n=1000, random_state=42)  # Sample 1,000 rows for example

# Preprocessing steps
df.columns = df.columns.str.lower()
df['starttime'] = pd.to_datetime(df['starttime'])
df['state'] = df['state'].fillna(value='CON')
df['sport'] = df['sport'].ffill()
df['dport'] = df['dport'].ffill()
df['stos'] = df['stos'].fillna(value=0.0)
df['dtos'] = df['dtos'].fillna(value=0.0)
df = df.drop(df[df['stos'] == 192.0].index)

train, test = train_test_split(df, test_size=0.3, random_state=42)

train_0, train_1 = train['label'].value_counts().iloc[0] / len(train.index), train['label'].value_counts().iloc[1] / len(train.index)
test_0, test_1 = test['label'].value_counts().iloc[0] / len(test.index), test['label'].value_counts().iloc[1] / len(test.index)
print("In Train: there are {} % of class 0 and {} % of class 1".format(train_0, train_1))
print("In Test: there are {} % of class 0 and {} % of class 1".format(test_0, test_1))

#  Utility function

def multi_corr(col1, col2="target", df=train):
    '''
    This function returns correlation between 2 given features.
    Also gives corr of the given features with "label" after applying log1p to it.
    '''
    corr = df[[col1, col2]].corr().iloc[0,1]
    log_corr = df[col1].apply(np.log1p).corr(df[col2])

    print("Correlation : {}\nlog_Correlation: {}".format(corr, log_corr))
def corr(col1, col2="target", df=train):
    """
    This function returns correlation between 2 given features
    """
    return df[[col1, col2]].corr().iloc[0,1]
def convertToOneClass(y):
    if y == 1:
        return -1
    return 1
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

train.drop(columns='Unnamed: 0', axis=1, inplace=True)
test.drop(columns='Unnamed: 0', axis=1, inplace=True)
print(train.shape, test.shape)
print(train)
saved_dict = {}
print(train.columns)
starttime = train['starttime']
srcaddr = train['srcaddr']
dstaddr = train['dstaddr']
state = train['state']
# Dropping columns which are not useful for the classification
# label is for binary classification
# all the other columns are address related and not present in sample train data
train.drop(['starttime', 'srcaddr', 'dstaddr', 'state'], axis=1, inplace=True)
# To use during test data transformation
saved_dict['to_drop'] = ['starttime', 'srcaddr', 'dstaddr', 'state']
print(train.shape, test.shape)
mode_dict = train.mode().iloc[0].to_dict()
print(mode_dict)
saved_dict['moded_featute'] = mode_dict
print(saved_dict)
# creating x and y set from the dataset
x_train, y_train, y_train_label = train.drop(columns=['label', 'target']), train['target'], train['label']
# x_test, y_test, y_test_label = test.drop(columns=['label', 'target']), test['target'], test['label']
print(x_train.shape, y_train.shape, y_train_label.shape)
print()
# print(x_test.shape, y_test.shape, y_test_label.shape)
(9950992, 10) (9950992,) (9950992,)

# Saving all the files to disk to use later
pickle.dump((x_train, y_train, y_train_label), open('./final_train.pkl', 'wb'))
# pickle.dump((x_test, y_test, y_test_label), open('./final_test.pkl', 'wb'))
# getting categorical and numerical columns in 2 diff lists
num_col = [ 'dur', 'sport', 'dport', 'totpkts', 'totbytes', 'srcbytes']
cat_col = list(set(x_train.columns) - set(num_col))
# To use later, during test data cleaning
saved_dict['cat_col'] = cat_col
saved_dict['num_col'] = num_col
print(saved_dict)
def is_hex(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

def convertHexInSport(port):
    if str(int(port, 16)) != port:
        return int(port, 16)

    return int(port)
x_train['sport'] = x_train['sport'].apply(convertHexInSport)
x_train['dport'] = x_train['dport'].apply(convertHexInSport)