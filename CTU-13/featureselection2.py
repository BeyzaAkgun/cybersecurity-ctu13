import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier  # Add this import
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
df = pd.read_csv(binetflow_file)

# Display basic information about the dataset
print(df.info())
print(df.dtypes)

# Handle missing values

df['State'] = df['State'].fillna('CON')
df['Sport'] = df['Sport'].fillna(method='pad')
df['Dport'] = df['Dport'].fillna(method='pad')
df['sTos'] = df['sTos'].fillna(0.0)
df['dTos'] = df['dTos'].fillna(0.0)

# Convert 'StartTime' to datetime
df['StartTime'] = pd.to_datetime(df['StartTime'])

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Verify the missing values are handled
print(df.isnull().sum())

selected_features_5 = ['Dport', 'TotPkts', 'TotBytes', 'Dur', 'SrcBytes']
selected_features_6 = selected_features_5 + ['sTos']  # Adding another feature for the 6-feature subset
selected_features_7 = selected_features_6 + ['Dir']  # Adding another feature for the 7-feature subset
# Define feature subsets
feature_subsets = {
    5: selected_features_5,
    6: selected_features_6,
    7: selected_features_7
}

# Split data into features (X) and target variable (y)
X = df[selected_features_5]  # Change this to select different feature subsets
y = df['Label']
# print(X)
# print(y)
X = pd.get_dummies(df[selected_features_5])

# Split data into training and testing sets


# # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Evaluation
# def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))

# Decision Tree
print("Decision Tree with 5-feature subset:")
dt_5 = DecisionTreeClassifier()
# train_and_evaluate_model(dt_5, X_train, X_test, y_train, y_test)
dt_5.fit(X_train,y_train)
y_pred=dt_5.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# # Random Forest
# print("Random Forest with 5-feature subset:")
# rf_5 = RandomForestClassifier(n_estimators=10)  # m=10 as mentioned in the text
# train_and_evaluate_model(rf_5, X_train, X_test, y_train, y_test)

# # K-Nearest Neighbors
# print("K-Nearest Neighbors with 5-feature subset:")
# knn_5 = KNeighborsClassifier(n_neighbors=1)  # k=1 as mentioned in the text
# train_and_evaluate_model(knn_5, X_train, X_test, y_train, y_test)