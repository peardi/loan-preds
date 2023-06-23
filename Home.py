import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder,RobustScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier
from session_state import SessionState



#🛑 Code to set the Dashboard format to wide (the content will fill the entire width of the page instead of having wide margins)
def do_stuff_on_page_load():
    st.set_page_config(layout="wide")
do_stuff_on_page_load()

#Set Header
#🛑 Code to set the header
st.header('Model metrics v1', anchor=None)



#🛑 Code to import the dataset
df = pd.read_csv('https://miles-become-a-data-scientist.s3.us-east-2.amazonaws.com/J3/M3/data/train.csv')

st.session_state['df'] = df 
#🛑 Code to persist the DataFrame between pages of the same Dashboard. Without this, any other page would need to re import the DataFrame and save it to df again.
#st.session_state['df'] = df 

#Adding a random_state seed for you to remember to use it!
RANDOM_STATE=42

# 1
df = df.loc[df.AMT_INCOME_TOTAL < 9000000.0]
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = df['TARGET']


X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=RANDOM_STATE)

# Transform 'DAYS_BIRTH' variable to a better understanding: Days -> Years
X_train['AGE'] = -(X_train['DAYS_BIRTH'] / 365).astype(int)
X_train.drop('DAYS_BIRTH', axis=1, inplace=True)

X_test['AGE'] = -(X_test['DAYS_BIRTH'] / 365).astype(int)
X_test.drop('DAYS_BIRTH', axis=1, inplace=True)

# Proportion of client's income allocated to loan repayment. Higher ratio indicates higher financial burden
X_train['ANNUITY_INCOME_RATIO'] = X_train['AMT_ANNUITY'] / X_train['AMT_INCOME_TOTAL']
X_test['ANNUITY_INCOME_RATIO'] = X_test['AMT_ANNUITY'] / X_test['AMT_INCOME_TOTAL']

# Proportion of client's income allocated to loan repayment. Higher ratio indicates higher financial burden
X_train['children_RATIO'] = X_train['CNT_CHILDREN'] / X_train['CNT_FAM_MEMBERS']
X_test['children_RATIO'] = X_test['CNT_CHILDREN'] / X_test['CNT_FAM_MEMBERS']

# The code replaces the 'OWN_CAR_AGE' column with a binary feature 'OWN_CAR_IND', indicating car ownership status.
X_train['OWN_CAR_IND'] = X_train['OWN_CAR_AGE'].notnull().astype(int)
X_train.drop('OWN_CAR_AGE', axis=1, inplace=True)

X_test['OWN_CAR_IND'] = X_test['OWN_CAR_AGE'].notnull().astype(int)
X_test.drop('OWN_CAR_AGE', axis=1, inplace=True)

# Calculate the ratio of AMT_ANNUITY to AMT_GOODS_PRICE
X_train['ANNUITY_GOODS_PRICE_RATIO'] = X_train['AMT_ANNUITY'] / X_train['AMT_GOODS_PRICE']
X_test['ANNUITY_GOODS_PRICE_RATIO'] = X_test['AMT_ANNUITY'] / X_test['AMT_GOODS_PRICE']

# Calculate the ratio of AMT_ANNUITY to AMT_CREDIT
X_train['ANNUITY_CREDIT_RATIO'] = X_train['AMT_ANNUITY'] / X_train['AMT_CREDIT']
X_test['ANNUITY_CREDIT_RATIO'] = X_test['AMT_ANNUITY'] / X_test['AMT_CREDIT']

num_features = [
    'AMT_ANNUITY',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'CNT_CHILDREN',
    'CNT_FAM_MEMBERS',
    'DAYS_EMPLOYED',
    'DAYS_ID_PUBLISH',
    'DAYS_LAST_PHONE_CHANGE',
    'DAYS_REGISTRATION',
    'AGE',
    'ANNUITY_INCOME_RATIO',
    'children_RATIO',
    'ANNUITY_GOODS_PRICE_RATIO',
    'ANNUITY_CREDIT_RATIO'
]

cat_features = [
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE',
    'OCCUPATION_TYPE',
    'FLAG_EMAIL',
    'FLAG_PHONE',
    'FLAG_WORK_PHONE',
    'OWN_CAR_IND'
]

for col in num_features:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

imputer_num = SimpleImputer(strategy='mean')
imputer_cat = SimpleImputer(strategy='constant', fill_value='missing')

scaler_robust = RobustScaler()
scaler_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Handle missing or null values in numeric columns
X_train[num_features] = X_train[num_features].fillna(0)
X_test[num_features] = X_test[num_features].fillna(0)

X_numeric_imputed_train = imputer_num.fit_transform(X_train[num_features])
X_numeric_imputed_test = imputer_num.transform(X_test[num_features])

# Convert numeric columns containing strings to NaN
X_numeric_imputed_train = pd.DataFrame(X_numeric_imputed_train, columns=num_features)
X_numeric_imputed_test = pd.DataFrame(X_numeric_imputed_test, columns=num_features)

X_numeric_imputed_train = X_numeric_imputed_train.apply(pd.to_numeric, errors='coerce')
X_numeric_imputed_test = X_numeric_imputed_test.apply(pd.to_numeric, errors='coerce')

x_numeric_scaled_train = scaler_robust.fit_transform(X_numeric_imputed_train)
x_numeric_scaled_test = scaler_robust.transform(X_numeric_imputed_test)

# Handle NaN values as needed
x_numeric_scaled_train[np.isnan(x_numeric_scaled_train)] = 0
x_numeric_scaled_test[np.isnan(x_numeric_scaled_test)] = 0

X_categorical_imputed_train = imputer_cat.fit_transform(X_train[cat_features])
X_categorical_imputed_test = imputer_cat.transform(X_test[cat_features])

x_categorical_scaled_train = scaler_ohe.fit_transform(X_categorical_imputed_train)
x_categorical_scaled_test = scaler_ohe.transform(X_categorical_imputed_test)

X_preprocessed_train = np.hstack([x_numeric_scaled_train, x_categorical_scaled_train])
X_preprocessed_test = np.hstack([x_numeric_scaled_test, x_categorical_scaled_test])
feature_names = num_features + list(scaler_ohe.get_feature_names_out(cat_features))

X_train_proc_dff = pd.DataFrame(X_preprocessed_train, columns=feature_names)
X_test_proc_dff = pd.DataFrame(X_preprocessed_test, columns=feature_names)

#model_train
s = {'class_weight': 'balanced', 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': -1, 'min_child_samples': 20, 'n_estimators': 200, 'num_leaves': 31, 'subsample': 0.8}
opt_model = LGBMClassifier(**s, random_state=RANDOM_STATE, n_jobs=-1).fit(X_train_proc_dff,y_train)

#model predictions for the positive class (1)
model_preds = opt_model.predict_proba(X_test_proc_dff)[:,1]

#true values of target
model_true = y_test

# Assuming you have the true labels for the test dataset in 'model_true' and predicted probabilities in 'model_preds'

# Calculate predicted labels based on the threshold
threshold = 0.71
model_pred_labels = [1 if prob >= threshold else 0 for prob in model_preds]

# Create a SessionState object
session_state = SessionState.get(average_cost=0)
# Create a DataFrame with the true labels and predicted labels
results_df = pd.DataFrame({'True Labels': model_true, 'Predicted Labels': model_pred_labels})

# Filter the DataFrame to include only the positive predictions
positive_predictions = results_df[results_df['Predicted Labels'] == 1]

# Calculate the average cost on the positive predictions
average_cost = positive_predictions['AMT_CREDIT'].mean()
session_state.average_cost = average_cost


# Calculate evaluation metrics
accuracy = accuracy_score(model_true, model_pred_labels)
precision = precision_score(model_true, model_pred_labels)
recall = recall_score(model_true, model_pred_labels)
f1 = f1_score(model_true, model_pred_labels)

# Calculate ROC curve
fpr, tpr, thresholds_roc = roc_curve(model_true, model_preds)

# Calculate precision-recall curve
precision_pr, recall_pr, thresholds_pr = precision_recall_curve(model_true, model_preds)

# Calculate confusion matrix
confusion_mat = confusion_matrix(model_true, model_pred_labels)


num_applicants = df.size
default_count = df[df['TARGET'] == 1].shape[0]
repay_count = df[df['TARGET'] == 0].shape[0]
default_rate = ((default_count/num_applicants)*100)


st.sidebar.header('Light Overview')

st.sidebar.subheader('Number of applicants')
st.sidebar.write(num_applicants)

default_rate = (default_rate, '%')
st.sidebar.subheader('Probability of default')
st.sidebar.write(default_rate)

