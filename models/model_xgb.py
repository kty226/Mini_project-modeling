from tkinter import Grid
from lightgbm import early_stopping
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn import preprocessing
import pandas as pd


def encode_FE(df1, df2):
    for col in df1.columns:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')
    return df1, df2

def XGBoost(train, test, submission):

    y_train = train['isFraud'].copy()
    # Drop target, fill in NaNs
    X_train = train.drop('isFraud', axis=1)
    # X_train = train.copy()
    X_test = test.copy()

    del train, test

    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)


    X_train, X_test = encode_FE(X_train, X_test)

    clf = xgb.XGBClassifier(
            use_label_encoder = False,
            eval_metric = 'logloss',
            n_estimators=500,
            max_depth=9,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            missing=-999,
            random_state=42,
            verbosity = 1
            # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
    )




    clf.fit(X_train, y_train)


    submission['isFraud'] = clf.predict_proba(X_test)[:,1]
    submission.to_csv('xgb.csv')


# %%
