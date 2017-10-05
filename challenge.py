# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline 

# Preprocessing
df_train = pd.read_csv("E:/Kaggle/Seguro/train.csv", header=0)
df_test = pd.read_csv("E:/Kaggle/Seguro/test.csv", header=0)

X_train = df_train.iloc[:,2:]
y_train = df_train.iloc[:,1]
X_test = df_test.iloc[:,1:]

X = pd.concat([X_train, X_test])
X = X.replace(-1, np.NaN)
X.drop(['ps_car_03_cat', 'ps_car_05_cat', 'ps_reg_03','ps_car_14'], axis=1, inplace=True)

# Imputing by most frequent
for col in X.columns[X.isnull().sum()>0]:
    X[col] = X[col].transform(lambda x:x.fillna(x.value_counts().idxmax()))

# Reshaping	
X_train_ = X.iloc[:X_train.shape[0],:]
X_test_ = X.iloc[X_train.shape[0]:,:]

# Categorical features
cat_features_list = []
for i in range(X.shape[1]):
    if 'cat' in X.columns[i]:
        cat_features_list.append(i)

# Initialize CatBoostClassifier
model = CatBoostClassifier(verbose=True)
# Fit model
model.fit(X_train_, y_train, cat_features = cat_features_list, verbose=True)
# Get predicted classes
preds_class = model.predict(X_test)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(X_test)
# Saving
a = np.reshape((preds_proba[:,1]), (preds_proba.shape[0],1))
df_pred = pd.read_csv("E:/Kaggle/Seguro/sample_submission.csv", header=0)
df_pred['target']=a
df_pred.to_csv("E:/Kaggle/Seguro/prediction.csv",index=False)

		
