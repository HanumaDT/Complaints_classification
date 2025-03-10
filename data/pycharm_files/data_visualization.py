import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
label_encoder = LabelEncoder()

data_path = "C:/Users/Ravi Kiran/Downloads/dataset/case_study_data.csv"
df = pd.read_csv(data_path)
# print(df.head())

# print(len(df))

# print(df.info())

# print(df['product_group'].nunique())

# print(df['product_group'].describe())

# print(df['product_group'].unique())

# pd.reset_option('display.max_colwidth')
df_bank_service = df[df['product_group'] == 'bank_service']
# print(df_bank_service['text'].head(5))

df['product_group_numeric'] = label_encoder.fit_transform(df['product_group'])
# print(df.head())

X = df['text']
y = df['product_group_numeric']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train[0:5])
print(y_train[0:5])
# print(len(X_train))
# print(len(X_test))


