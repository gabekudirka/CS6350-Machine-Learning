import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

df_train = pd.read_csv('train_final.csv')
df_test = pd.read_csv('test_final.csv')
df_test = df_test.drop('ID',axis = 1)

for col in ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'sex', 'race', 'native.country']:
    one_hot_train = pd.get_dummies(df_train[col], prefix=col)
    one_hot_test = pd.get_dummies(df_test[col], prefix=col)

    df_train = df_train.drop(col,axis = 1)
    df_train = pd.concat([df_train,one_hot_train],axis=1)

    df_test = df_test.drop(col,axis = 1)
    df_test = pd.concat([df_test,one_hot_test],axis=1)

missing_col = None
for col in df_test.columns:
    if col not in df_train.columns:
        df_train[col] = 0
        print(col)

X = df_train.loc[:, df_train.columns != 'income>50K']
y = df_train['income>50K']

X_test_final = df_test.loc[:, df_test.columns != 'income>50K']

estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)
parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05],
    "subsample":[0.5, 0.75, 1],
    "min_child_weight":[1,5,15]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)

grid_search.fit(X,y)

print(" Results from Grid Search " )
print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_search.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)