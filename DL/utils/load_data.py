from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml

import numpy as np
import pandas as pd

def load_data(args):
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    num_idx = []
    args.cat_dims = []

    if args.dataset == 'heart_failure':
        heart_failure = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
        X = heart_failure.drop('DEATH_EVENT', axis=1)
        y = heart_failure['DEATH_EVENT']
        # Exclude rows with missing values
        combine = X.copy()
        combine['target'] = y
        missing_values = pd.isnull(combine).sum(axis=1)
        X = X[missing_values == 0]
        y = y[missing_values == 0]

        for column in X.columns:
            X[column] = scaler.fit_transform(X[column].values.reshape(-1, 1))

        X = X.values
        y = label_encoder.fit_transform(y)

    elif args.dataset == 'parkinsons':
        parkinsons = pd.read_csv('data/parkinsons.data')
        X = parkinsons.drop(columns=['name', 'status'], axis=1)
        y = parkinsons['status']
        # Exclude rows with missing values
        combine = X.copy()
        combine['target'] = y
        missing_values = pd.isnull(combine).sum(axis=1)
        X = X[missing_values == 0]
        y = y[missing_values == 0]

        for column in X.columns:
            X[column] = scaler.fit_transform(X[column].values.reshape(-1, 1))

        X = X.values
        y = label_encoder.fit_transform(y)

    else:
        X,y = fetch_openml(args.dataset, version=1, return_X_y=True, as_frame=True)

        # Exclude rows with missing values
        combine = X.copy()
        combine['target'] = y
        missing_values = pd.isnull(combine).sum(axis=1)
        X = X[missing_values == 0]
        y = y[missing_values == 0]

        for column in X.columns:
            X[column] = scaler.fit_transform(X[column].values.reshape(-1, 1))

        X = X.values
        y = label_encoder.fit_transform(y)

    return X,y
