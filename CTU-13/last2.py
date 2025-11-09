
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load the dataset
binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
df = pd.read_csv(binetflow_file)

# numpy.core._exceptions._ArrayMemoryError: Unable to allocate 194. GiB for an array with shape (73786, 2824636) and data type bool
# Display basic information about the dataset
print(df.info())



print(df.dtypes)
missing_values = df.isnull().sum()
print(missing_values)

sport_count = df['Sport'].count()
print("Number of rows in 'Sport' column:", sport_count)
Dport_count = df['Dport'].count()
print("Number of rows in 'Dport' column:", Dport_count)
sTos_count = df['sTos'].count()
print("Number of rows in 'sTos' column:", sTos_count)
dTos_count = df['dTos'].count()
print("Number of rows in 'dTos' column:", dTos_count)

sport_percentage = (9379 / sport_count) * 100
dport_percentage = (4390 / Dport_count) * 100
sTos_percentage = (10590 / sTos_count) * 100
dTos_percentage = (195190 / dTos_count) * 100

# print("Missing Sport values are: {:.2f}%".format(sport_percentage))
# print("Missing Dport values are: {:.2f}%".format(dport_percentage))
# print("Missing sTos values are: {:.2f}%".format(sTos_percentage))
# print("Missing dTos values are: {:.2f}%".format(dTos_percentage))
# print("State is already one row")

# Handle missing values
df['State'] = df['State'].fillna('CON')
df['Sport'] = df['Sport'].ffill()
df['Dport'] = df['Dport'].ffill()
df['sTos'] = df['sTos'].fillna(0.0)
df['dTos'] = df['dTos'].fillna(0.0)

# Convert 'StartTime' to datetime
df['StartTime'] = pd.to_datetime(df['StartTime'])

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# # Verify the missing values are handled
# print(df.isnull().sum())
 
df = df.sample(n=2000, random_state=42)  # Sample 1,000 rows for example
# Define feature subsets
selected_features_5 = ['Dport', 'TotPkts', 'TotBytes', 'Dur', 'SrcBytes']
selected_features_6 = selected_features_5 + ['sTos']  # Adding another feature for the 6-feature subset
selected_features_7 = selected_features_6 + ['Dir']  # Adding another feature for the 7-feature subset

# # Feature subsets dictionary
feature_subsets = {
    5: selected_features_5,
    6: selected_features_6,
    7: selected_features_7
}

# Model Training and Evaluation
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))
    return f1_score(y_test, y_pred, average='weighted')

# Iterate over each feature subset and evaluate models
for num_features, features in feature_subsets.items():
    print(f"\nEvaluating models with {num_features}-feature subset:")

    # Select features and target
    X = pd.get_dummies(df[features])
    y = df['Label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
    print(f"\nDecision Tree with {num_features}-feature subset:")
    dt = DecisionTreeClassifier()
    dt_f1 = train_and_evaluate_model(dt, X_train, X_test, y_train, y_test)
    print(f"Decision Tree F1 Score: {dt_f1:.4f}")

    # Random Forest
    print(f"\nRandom Forest with {num_features}-feature subset:")
    rf = RandomForestClassifier(n_estimators=10)
    rf_f1 = train_and_evaluate_model(rf, X_train, X_test, y_train, y_test)
    print(f"Random Forest F1 Score: {rf_f1:.4f}")

    # K-Nearest Neighbors
    print(f"\nK-Nearest Neighbors with {num_features}-feature subset:")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn_f1 = train_and_evaluate_model(knn, X_train, X_test, y_train, y_test)
    print(f"K-Nearest Neighbors F1 Score: {knn_f1:.4f}")


