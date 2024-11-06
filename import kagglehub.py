# %%
import os
import subprocess
import pandas as pd
import numpy as np
import zipfile

# %%
# Download the dataset using subprocess
#subprocess.run(["kaggle", "datasets", "download", "rodsaldanha/arketing-campaign"])


# # %%

# # Extract the zip file to a folder named "data"
# with zipfile.ZipFile("arketing-campaign.zip", 'r') as zip_ref:
#     zip_ref.extractall("data")

# print(os.listdir("data/"))  # Check files in the 'data' directory


# # Load the dataset (replace 'marketing_campaign.csv' with the actual filename)
# df = pd.read_csv("arketing_campaign.csv")

# # Display the first few rows of the dataset
# print(df.head())
 
# %%

# Load the CSV file with a semicolon delimiter
df = pd.read_csv("data/marketing_campaign.csv", delimiter=';')


# %%
#view dataset
df.head()

# View the structure of the dataset
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Summary statistics for numerical columns
print(df.describe())

# %%
# Fill missing Income values with the median
df['Income'].fillna(df['Income'].median(), inplace=True)

# %%
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Tenure'] = (pd.to_datetime('today') - df['Dt_Customer']).dt.days

# %%
df.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

# %%
df['Age'] = pd.to_datetime('today').year - df['Year_Birth']
df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
df['Total_Accepted_Campaigns'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Age Distribution
sns.histplot(df['Age'], bins=20)
plt.title('Customer Age Distribution')
plt.show()

# Income Distribution
sns.histplot(df['Income'], bins=20)
plt.title('Income Distribution')
plt.show()

# Total Spending Distribution
sns.histplot(df['Total_Spending'], bins=20)
plt.title('Total Spending Distribution')
plt.show()

# Total Accepted Campaigns Distribution
sns.countplot(df['Total_Accepted_Campaigns'])
plt.title('Distribution of Total Accepted Campaigns')
plt.show()

# %%
print("Maximum Total Spending:", df['Total_Spending'].max())

# %%
# Define spending levels
spending_bins = [0, 500, 1500, df['Total_Spending'].max()]
spending_labels = ['Low', 'Medium', 'High']
df['Spending_Segment'] = pd.cut(df['Total_Spending'], bins=spending_bins, labels=spending_labels)

sns.countplot(data=df, x='Spending_Segment')
plt.title('Spending Segments')
plt.show()

# %%
accepted_df = df[df['Total_Accepted_Campaigns'] > 0]
print("Average income of customers who accepted campaigns:", accepted_df['Income'].mean())
print("Average age of customers who accepted campaigns:", accepted_df['Age'].mean())

# %%

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier
from xgboost import XGBClassifier

# Assume df is already preprocessed and 'Response' is the target variable
# Features used for prediction
features = ['Age', 'Income', 'Recency', 'Total_Spending', 'Total_Accepted_Campaigns']
X = df[features]
y = df['Response']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 1: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 2: Try Different Classifiers
# 2a. Random Forest Classifier with Grid Search for Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)
grid_search_rf.fit(X_train_resampled, y_train_resampled)

# Best Random Forest model after Grid Search
best_rf = grid_search_rf.best_estimator_

# 2b. XGBoost Classifier
xgb = XGBClassifier(random_state=42, scale_pos_weight=3)  # scale_pos_weight helps with imbalance in XGBoost
xgb.fit(X_train_resampled, y_train_resampled)

# 2c. Balanced Bagging Classifier
bbc = BalancedBaggingClassifier(random_state=42)
bbc.fit(X_train, y_train)  # BalancedBaggingClassifier handles imbalance internally

# Step 3: Model Evaluation
# Evaluate Random Forest
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]))

# Evaluate XGBoost
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("AUC-ROC:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

# Evaluate Balanced Bagging Classifier
y_pred_bbc = bbc.predict(X_test)
print("\nBalanced Bagging Classifier Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_bbc))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bbc))
print("Classification Report:\n", classification_report(y_test, y_pred_bbc))
print("AUC-ROC:", roc_auc_score(y_test, bbc.predict_proba(X_test)[:, 1]))

# Step 4: Choose the Best Model Based on AUC-ROC and F1 Score


# %%
