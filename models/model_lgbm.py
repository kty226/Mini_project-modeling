import time
from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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


def lgbm_val(train, test):

    y_train = train['isFraud'].copy()
    # Drop target, fill in NaNs
    X_train = train.drop('isFraud', axis=1)
    # X_train = train.copy()
    X_test = test.copy()

    del train, test

    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)



    X_train, X_test = encode_FE(X_train, X_test)

    start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.3, random_state=42)

    clf = LGBMClassifier(
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.05,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    predicted=clf.predict(X_test)
    print('Classification of the result is:')
    print(accuracy_score(y_test, predicted))
    end = time.time()
    print('Execution item is :')
    print(end - start)



def lgbm(train, test, submission):

    y_train = train['isFraud'].copy()
    # Drop target, fill in NaNs 
    X_train = train.drop('isFraud', axis=1)
    # X_train = train.copy()
    X_test = test.copy()

    del train, test

    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)



    X_train, X_test = encode_FE(X_train, X_test)

    start = time.time()

    clf = LGBMClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        random_state=42,
    )

    clf.fit(X_train, y_train)

    end = time.time()

    print(end - start)
    submission['isFraud'] = clf.predict_proba(X_test)[:,1]
    submission.to_csv('lgbm.csv')



