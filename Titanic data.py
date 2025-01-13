import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io
import os

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
    print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
    with zipfile.ZipFile(io.BytesIO(uploaded[fn]), 'r') as zip_ref:
        zip_ref.extractall('/content/')

extracted_files = os.listdir('/content/')
print(f"Extracted files: {extracted_files}")

df = pd.read_csv('/content/train.csv') 

print("Missing values before cleaning:")
print(df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop('Cabin', axis=1, inplace=True)

print("Missing values after cleaning:")
print(df.isnull().sum())

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


print("Summary statistics:")
print(df.describe())

plt.figure(figsize=(8, 6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Sex_male', hue='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=20, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(10,8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print(df.head())
