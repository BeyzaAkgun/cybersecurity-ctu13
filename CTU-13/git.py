import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
df = pd.read_csv(binetflow_file)

df.columns = df.columns.str.lower()

df['starttime'] = pd.to_datetime(df['starttime'])

    # We replace Null value to 'CON'
df['state'] = df.state.fillna(value='CON')

    # fill nan port forward 
df['sport'] = df.sport.fillna(method='pad')
df['dport'] = df.dport.fillna(method='pad')

df['stos'] = df.stos.fillna(value=0.0)
df['dtos'] = df.dtos.fillna(value=0.0)

df = df.drop(df[df.stos == 192.0].index)
train, test = train_test_split(df, test_size=0.3, random_state=42)

train_0, train_1 = train['label'].value_counts(normalize=True)
test_0, test_1 = test['label'].value_counts(normalize=True)

print("In Train: there are {} % of class 0 and {} % of class 1".format(train_0 * 100, train_1 * 100))
print("In Test: there are {} % of class 0 and {} % of class 1".format(test_0 * 100, test_1 * 100))

def multi_corr(col1, col2="target", df=train):
    corr = df[[col1, col2]].corr().iloc[0, 1]
    log_corr = df[col1].apply(np.log1p).corr(df[col2])
    print("Correlation : {}\nlog_Correlation: {}".format(corr, log_corr))

def corr(col1, col2="target", df=train):
    return df[[col1, col2]].corr().iloc[0, 1]

def convertToOneClass(y):
    return -1 if y == 1 else 1

# Dropping unnecessary columns
train.drop(columns=['Unnamed: 0', 'starttime', 'srcaddr', 'dstaddr', 'state'], axis=1, inplace=True)
test.drop(columns=['Unnamed: 0', 'starttime', 'srcaddr', 'dstaddr', 'state'], axis=1, inplace=True)

# Saving dropped columns for future reference
saved_dict = {'to_drop': ['starttime', 'srcaddr', 'dstaddr', 'state']}

# Most frequent values in columns (mode)
mode_dict = train.mode().iloc[0].to_dict()
saved_dict['moded_featute'] = mode_dict

# Separating features and labels
x_train, y_train, y_train_label = train.drop(columns=['label', 'target']), train['target'], train['label']
num_col = ['dur', 'sport', 'dport', 'totpkts', 'totbytes', 'srcbytes']
cat_col = list(set(x_train.columns) - set(num_col))

saved_dict['cat_col'] = cat_col
saved_dict['num_col'] = num_col

def is_hex(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

def convertHexInSport(port):
    return int(port, 16) if is_hex(port) else int(port)

x_train['sport'] = x_train['sport'].apply(convertHexInSport)
x_train['dport'] = x_train['dport'].apply(convertHexInSport)

scaler = MinMaxScaler()
scaler = scaler.fit(x_train[num_col])
x_train[num_col] = scaler.transform(x_train[num_col])

# OneHot Encoding
stos_ = OneHotEncoder()
dtos_ = OneHotEncoder()
dir_ = OneHotEncoder()
proto_ = OneHotEncoder()

ohe_stos = stos_.fit(x_train.stos.values.reshape(-1, 1))
ohe_dtos = dtos_.fit(x_train.dtos.values.reshape(-1, 1))
ohe_dir = dir_.fit(x_train.dir.values.reshape(-1, 1))
ohe_proto = proto_.fit(x_train.proto.values.reshape(-1, 1))

for col, encoding in zip(['stos', 'dtos', 'dir', 'proto'], [ohe_stos, ohe_dtos, ohe_dir, ohe_proto]):
    x = encoding.transform(x_train[col].values.reshape(-1, 1))
    tmp_df = pd.DataFrame(x.toarray(), dtype='int64', columns=[col+'_'+str(i) for i in encoding.categories_[0]])
    x_train = pd.concat([x_train.drop(col, axis=1), tmp_df], axis=1)
   




