import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'
df = pd.read_csv(binetflow_file)

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

# Feature selection using Decision Tree with Gini importance
X = df.drop(columns=['Label'])  # Features
y = df['Label']  # Target variable
dt = DecisionTreeClassifier()
dt.fit(X, y)
feature_importances = dt.feature_importances_
selected_features = X.columns[np.argsort(feature_importances)[::-1]][:7]  # Select top 7 features

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[selected_features], df['Label'], test_size=0.2, random_state=42)

# Models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'k-NN': KNeighborsClassifier()
}

# Compare models
for model_name, model in models.items():
    print(f"Model: {model_name}")
    for n_features in [5, 6, 7]:
        print(f"Number of features: {n_features}")
        model.fit(X_train.iloc[:, :n_features], y_train)
        y_pred = model.predict(X_test.iloc[:, :n_features])
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))