# import pandas as pd
# import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns

# import os
# import pickle
# import datetime
# import itertools

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# # Load the dataset
# binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
# df = pd.read_csv(binetflow_file)

# Display basic information about the dataset
# print(df.info())



# print(df.dtypes)
# missing_values = df.isnull().sum()
# print(missing_values)

# sport_count = df['Sport'].count()
# print("Number of rows in 'Sport' column:", sport_count)
# Dport_count = df['Dport'].count()
# print("Number of rows in 'Dport' column:", Dport_count)
# sTos_count = df['sTos'].count()
# print("Number of rows in 'sTos' column:", sTos_count)
# dTos_count = df['dTos'].count()
# print("Number of rows in 'dTos' column:", dTos_count)

# sport_percentage = (9379 / sport_count) * 100
# dport_percentage = (4390 / Dport_count) * 100
# sTos_percentage = (10590 / sTos_count) * 100
# dTos_percentage = (195190 / dTos_count) * 100

# print("Missing Sport values are: {:.2f}%".format(sport_percentage))
# print("Missing Dport values are: {:.2f}%".format(dport_percentage))
# print("Missing sTos values are: {:.2f}%".format(sTos_percentage))
# print("Missing dTos values are: {:.2f}%".format(dTos_percentage))
# print("State is already one row")
# print("If a column has more than 50% null values, it may be best to drop that column as it won't provide enough information for the model. However, if the number of null values is less than 50%, we can use a simple imputer to fill in the missing values with the mean, median, or most frequent value.")
    
# # df['Sport'].fillna('Unknown', inplace=True)
# df['Dport'].fillna('Unknown', inplace=True)
# df['sTos'].fillna(df['sTos'].mode()[0], inplace=True)
# df.drop('dTos', axis=1, inplace=True)
# Since 'State' has no missing values and you mentioned it's already one, you can drop it as you mentioned.
# df.drop('State', axis=1, inplace=True)



# data=df.copy()
# data.columns = data.columns.str.lower()

# data['target'] = data['label'].apply(convertLabel)
# data['starttime'] = pd.to_datetime(data['starttime'])

# # We replace Null value to Con 
# data['state'] = data.state.fillna(value='CON')

# # fill nan port forward 
# data['sport'] = data.sport.fillna(method='pad')
# data['dport'] = data.dport.fillna(method='pad')

# data['stos'] = data.stos.fillna(value=0.0)
# data['dtos'] = data.dtos.fillna(value=0.0)

# data = data.drop(data[data.stos == 192.0].index)


# print(df["State"].value_counts())


#CHATGPT
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix

# # Load the dataset
# binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
# df = pd.read_csv(binetflow_file)

# # Display basic information about the dataset
# print(df.info())
# print(df.dtypes)

# # Handle missing values

# df['State'] = df['State'].fillna('CON')
# df['Sport'] = df['Sport'].fillna(method='pad')
# df['Dport'] = df['Dport'].fillna(method='pad')
# df['sTos'] = df['sTos'].fillna(0.0)
# df['dTos'] = df['dTos'].fillna(0.0)

# # Convert 'StartTime' to datetime
# df['StartTime'] = pd.to_datetime(df['StartTime'])

# # Drop rows with any remaining missing values
# df.dropna(inplace=True)

# # Verify the missing values are handled
# print(df.isnull().sum())

# # Select relevant features based on the provided PDF
# # Here we choose a set of features, but this can be adjusted as necessary
# selected_features = ['Dur', 'Proto', 'SrcAddr', 'Sport', 'DstAddr', 'Dport', 'TotPkts', 'TotBytes', 'Label']

# # Extract the selected features
# df = df[selected_features]

# # Encode categorical variables
# df = pd.get_dummies(df, columns=['Proto', 'SrcAddr', 'Sport', 'DstAddr', 'Dport'])

# # Scale numerical features
# scaler = MinMaxScaler()
# df[['Dur', 'TotPkts', 'TotBytes']] = scaler.fit_transform(df[['Dur', 'TotPkts', 'TotBytes']])

# # Define the target variable and features
# X = df.drop('Label', axis=1)
# y = df['Label']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train Decision Tree Classifier
# dt_classifier = DecisionTreeClassifier()
# dt_classifier.fit(X_train, y_train)
# y_pred_dt = dt_classifier.predict(X_test)

# # Evaluate Decision Tree Classifier
# print("Decision Tree Classifier Report")
# print(classification_report(y_test, y_pred_dt))
# print("Confusion Matrix")
# print(confusion_matrix(y_test, y_pred_dt))

# # Train Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)
# y_pred_rf = rf_classifier.predict(X_test)

# # Evaluate Random Forest Classifier
# print("Random Forest Classifier Report")
# print(classification_report(y_test, y_pred_rf))
# print("Confusion Matrix")
# print(confusion_matrix(y_test, y_pred_rf))

# dPort, nPackets, vLen, mTime and sPort are most important
#1) Vector with ve features: [dPort, nPackets, nBytes,vLen, mLen
# 2) Vector with six features: [dPort, nPackets, nBytes,vLen, mLen, mTime]
#3) Vector with seven features: [dPort, nPackets, nBytes,vLen, mLen, mTime, vTime]
# These selected feature subsets were tested with the three
#  optimized models, DT, RF (m 
# 10) and k-NN (k 
# 1)
#  over the EQB-CTU13 dataset. Fig. 7 represents the weighted
#  120574
#  F1 scores, and shows a clear advantage of DT and RF over
#  k-NN. Moreover, k-NN also showed a higher computational
#  cost for classi cation than the other models (see Fig. 8).
#  Besides, it is expected that it will perform even worse in
#  computational time with larger datasets, since it searches for
#  the k mostsimilar samples amongallthetraining data. On the
#  other hand, while RF achieved higher F1 scores than DT


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier  # Add this import
# from sklearn.metrics import classification_report, confusion_matrix

# # Load the dataset
# binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
# df = pd.read_csv(binetflow_file)

# # Display basic information about the dataset
# print(df.info())
# print(df.dtypes)

# # Handle missing values

# df['State'] = df['State'].fillna('CON')
# df['Sport'] = df['Sport'].fillna(method='pad')
# df['Dport'] = df['Dport'].fillna(method='pad')
# df['sTos'] = df['sTos'].fillna(0.0)
# df['dTos'] = df['dTos'].fillna(0.0)

# # Convert 'StartTime' to datetime
# df['StartTime'] = pd.to_datetime(df['StartTime'])

# # Drop rows with any remaining missing values
# df.dropna(inplace=True)

# # Verify the missing values are handled
# print(df.isnull().sum())

# selected_features_5 = ['Dport', 'TotPkts', 'TotBytes', 'Dur', 'SrcBytes']
# selected_features_6 = selected_features_5 + ['sTos']  # Adding another feature for the 6-feature subset
# selected_features_7 = selected_features_6 + ['Dir']  # Adding another feature for the 7-feature subset
# # Define feature subsets
# feature_subsets = {
#     5: selected_features_5,
#     6: selected_features_6,
#     7: selected_features_7
# }

# # Split data into features (X) and target variable (y)
# X = df[selected_features_5]  # Change this to select different feature subsets
# y = df['Label']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model Training and Evaluation
# def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(classification_report(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))

# # Decision Tree
# print("Decision Tree with 5-feature subset:")
# dt_5 = DecisionTreeClassifier()
# train_and_evaluate_model(dt_5, X_train, X_test, y_train, y_test)

# # Random Forest
# print("Random Forest with 5-feature subset:")
# rf_5 = RandomForestClassifier(n_estimators=10)  # m=10 as mentioned in the text
# train_and_evaluate_model(rf_5, X_train, X_test, y_train, y_test)

# # K-Nearest Neighbors
# print("K-Nearest Neighbors with 5-feature subset:")
# knn_5 = KNeighborsClassifier(n_neighbors=1)  # k=1 as mentioned in the text
# train_and_evaluate_model(knn_5, X_train, X_test, y_train, y_test)



