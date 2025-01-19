import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, log_loss, confusion_matrix
from sklearn.datasets import fetch_openml, load_diabetes, load_breast_cancer
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# Set the seed for reproducibility
np.random.seed(42)
random.seed(42)
seed = 42

def load_dataset(dataset):
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    if dataset == 'heart_failure':
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

    elif dataset == 'parkinsons':
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
        X,y = fetch_openml(dataset, version=1, return_X_y=True, as_frame=True)

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

    return X,y, label_encoder

def save_model(model, path, algorithm_name):
    if algorithm_name == 'XGBoost' or algorithm_name == 'XGBoost_CPU':
        model.save_model(path + f'{algorithm_name}.json')
    elif algorithm_name == 'CatBoost':
        model.save_model(path + f'{algorithm_name}.cbm')
    else:
        joblib.dump(model, path + f'{algorithm_name}.pkl')

def csv_writer(path, algorithm_name, avg_scores_train, std_scores_train,
                avg_scores_test, std_scores_test, total_time):
    results_val = pd.DataFrame({
        'Model': [algorithm_name],
        'Accuracy': [f"{avg_scores_test['test_acc']} ± {std_scores_test['test_acc']}"],
        'F1 Score (Weighted)': [f"{avg_scores_test['test_f1_weighted']} ± {std_scores_test['test_f1_weighted']}"],
        'AUC Score': [f"{avg_scores_test['test_roc_auc']} ± {std_scores_test['test_roc_auc']}"],
        'Precision': [f"{avg_scores_test['test_precision']} ± {std_scores_test['test_precision']}"],
        'Recall': [f"{avg_scores_test['test_recall']} ± {std_scores_test['test_recall']}"],
        'Total Training Time': [total_time]
    })
    if not os.path.exists(path + f'val_results.csv'):
        results_val.to_csv(path + f'val_results.csv', index=False)
    else:
        results_val.to_csv(path + f'val_results.csv', mode='a', header=False, index=False)
    

    results_train = pd.DataFrame({
        'Model': [algorithm_name],
        'Accuracy': [f"{avg_scores_train['train_acc']} ± {std_scores_train['train_acc']}"],
        'F1 Score (Weighted)': [f"{avg_scores_train['train_f1_weighted']} ± {std_scores_train['train_f1_weighted']}"],
        'AUC Score': [f"{avg_scores_train['train_roc_auc']} ± {std_scores_train['train_roc_auc']}"],
        'Precision': [f"{avg_scores_train['train_precision']} ± {std_scores_train['train_precision']}"],
        'Recall': [f"{avg_scores_train['train_recall']} ± {std_scores_train['train_recall']}"],
        'Total Training Time': [total_time]
    })
    if not os.path.exists(path + f'train_results.csv'):
        results_train.to_csv(path + f'train_results.csv', index=False)
    else:
        results_train.to_csv(path + f'train_results.csv', mode='a', header=False, index=False)