import pyshark
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

binetflow_file = 'C:\\Users\\Asus\\Desktop\\capture20110810.binetflow.xz'

df = pd.read_csv(binetflow_file)

# for i in df.columns:
#    print(i)
#    print("Feature:", i)
#    print("Data Type:", df[i].dtype)
#    print("Sample Values:", df[i].unique())
#    print("//////////////////")

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# print(df.head())
# csv_file_path= 'C:\\Users\\Asus\\Desktop\\Ctu-13.Dataset-42.csv'

# df.to_csv('C:\\Users\\Asus\\Desktop\\Ctu-13.Dataset-42.csv', index=False)
# for column in df.columns:
#     first_five_rows = df[column].head()
#     print("First Five Rows of Column {}: \n{}".format(column, first_five_rows))
#     print("////////////////////////////////////")






# # Plotting network traffic patterns
# df = pd.read_csv(binetflow_file, nrows=10)
# plt.figure(figsize=(10, 6))
# df['SrcAddr'].value_counts().plot(kind='bar', figsize=(10, 6))
# plt.xlabel('Source IP Address')
# plt.ylabel('Number of Flows')
# plt.title('Number of Flows per Source IP Address')
# plt.show()

# # Plot histograms of selected features

# selected_features = ['TotPkts', 'TotBytes', 'Dur']
# df[selected_features].hist(figsize=(10, 6))
# plt.suptitle('Anomalies')
# plt.show()


# # Filter the dataset for normal and malicious traffic
# normal_traffic = df[df['Label'] == 'Normal']
# malicious_traffic = df[df['Label'] != 'Normal']

# # Plot box plots for selected features
# selected_features = ['TotPkts', 'TotBytes', 'Dur']
# plt.figure(figsize=(10, 6))
# for i, feature in enumerate(selected_features):
#     plt.subplot(1, len(selected_features), i+1)
#     plt.boxplot([normal_traffic[feature], malicious_traffic[feature]])
#     plt.xticks([1, 2], ['Normal', 'Malicious'])
#     plt.ylabel(feature)
# plt.suptitle('Box Plots of Selected Features for Normal and Malicious Traffic')
# plt.show()


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

#FEATURE SELECTİON
print(df.columns)

# Exclude non-numeric columns before calculating correlation
# numeric_df = df.select_dtypes(include=['number'])

# # Calculate Pearson correlation coefficients
# correlation_matrix = numeric_df.corr()

# # Assuming 'Label' is the target variable
# correlation_with_target = correlation_matrix[""].abs().sort_values(ascending=False)

# # Print correlation coefficients
# print("Pearson Correlation Coefficients with Target Variable:")
# print(correlation_with_target)