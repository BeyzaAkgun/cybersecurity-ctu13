import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

# Load the dataset
binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
df = pd.read_csv(binetflow_file)

# Preprocessing steps
df.columns = df.columns.str.lower()
df['starttime'] = pd.to_datetime(df['starttime'])
df['state'] = df['state'].fillna(value='CON')
df['sport'] = df['sport'].ffill()
df['dport'] = df['dport'].ffill()
df['stos'] = df['stos'].fillna(value=0.0)
df['dtos'] = df['dtos'].fillna(value=0.0)
df = df.drop(df[df['stos'] == 192.0].index)

# Drop rows with missing values in other columns
df = df.dropna()

# Feature Selection
# Separate features and target variable
X = df.drop(columns=['label'])
y = df['label']

# Encoding categorical variables
categorical_cols = ['proto', 'srcaddr', 'dir', 'dstaddr', 'state']
one_hot_encoder = OneHotEncoder(drop='first')
X_encoded = one_hot_encoder.fit_transform(X[categorical_cols])
X_encoded = pd.DataFrame(X_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
X.drop(columns=categorical_cols, inplace=True)
X.reset_index(drop=True, inplace=True)  # Resetting indices
X = pd.concat([X, X_encoded], axis=1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features
numerical_cols = ['dur', 'sport', 'dport', 'stos', 'dtos', 'totpkts', 'totbytes', 'srcbytes']
scaler = MinMaxScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Feature Selection using Information Gain
info_gain = mutual_info_classif(X_train, y_train)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(X.columns, info_gain, color='blue', alpha=0.7)
plt.xlabel('Information Gain')
plt.ylabel('Features')
plt.title('Feature Importance using Information Gain')
plt.show()

