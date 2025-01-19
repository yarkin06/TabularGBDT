import pandas as pd
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder,StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import joblib
import os
import time
import pickle
from tqdm import tqdm

from utils import csv_writer, load_dataset, save_model

# import warnings
# warnings.filterwarnings('ignore')


def main(args):

    # Set the seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    seed = 42

    if args.mode == 'data_prep':

        os.makedirs('ml_data', exist_ok=True)
        data_path = os.path.join('ml_data/')

        # Define the datasets
        datasets = {}
        datasets[args.data] = load_dataset(args.data)

        # Create a folder for each dataset
        for dataset_name in datasets.keys():
            print(f'\n\nRunning data preparation for {dataset_name}...')

            os.makedirs(data_path + dataset_name, exist_ok=True)
            data_data_path = os.path.join(data_path + dataset_name + '/')

            # Load the dataset
            dataset = datasets[dataset_name]
            X, y, label_encoder = dataset[0], dataset[1], dataset[2]
            print('Class names:', label_encoder.classes_)
            np.save(data_data_path + 'classes.npy', label_encoder.classes_)

            print('\nChecking for missing values...')
            print(pd.isnull(X).sum().sum())
            print(pd.isnull(y).sum().sum())

            class_names = label_encoder.classes_
            n_classes = len(class_names)

            print(f'\nX shape: {X.shape}')
            print(f'y shape: {y.shape}')
            print(f'Number of classes: {n_classes}\n')

            print('Save the entire dataset on one...')
            np.save(data_data_path + 'X.npy', X)
            np.save(data_data_path + 'y.npy', y)

    # Perform validation
    elif args.mode == 'train':
        dataset_name = args.data
        print(f'\nRunning validation for {dataset_name}...')

        os.makedirs('results/', exist_ok=True)
        results_path = os.path.join('results/')

        os.makedirs(results_path + 'ml_validation_results', exist_ok=True)
        os.makedirs(results_path + 'ml_models', exist_ok=True)

        data_path = os.path.join('ml_data/')
        val_path = os.path.join(results_path + 'ml_validation_results/')
        model_path = os.path.join(results_path + 'ml_models/')

        os.makedirs(val_path + dataset_name, exist_ok=True)
        os.makedirs(model_path + dataset_name, exist_ok=True)

        data_data_path = os.path.join(data_path + dataset_name + '/')
        data_val_path = os.path.join(val_path + dataset_name + '/')
        data_model_path = os.path.join(model_path + dataset_name + '/')

        X_train = np.load(data_data_path + 'X.npy', allow_pickle=True)
        y_train = np.load(data_data_path + 'y.npy', allow_pickle=True)

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load(data_data_path + 'classes.npy', allow_pickle=True)
        n_classes = len(label_encoder.classes_)

        if n_classes == 2:
            # Define the ML algorithms for binary classification
            algorithms = {
                'XGBoost': XGBClassifier(tree_method="gpu_hist", use_label_encoder=False, verbosity=0, eval_metric='logloss', objective='binary:logistic', seed=seed),
                'LogisticRegression': LogisticRegression(max_iter=10000000),
                'KNN': KNeighborsClassifier(),
                'RandomForest': RandomForestClassifier(),
                'DecisionTree': DecisionTreeClassifier(),
                'LDA': LinearDiscriminantAnalysis(),
                'SVM': SVC(probability=True),
                'LightGBM': LGBMClassifier(device='gpu', objective='binary', metric='binary_logloss', verbose=0, seed=seed),
                'CatBoost': CatBoostClassifier(task_type="GPU", eval_metric='Logloss', verbose=0, random_seed=seed),
            }
        else:
            # Define the ML algorithms for multi-class classification
            algorithms = {
                'XGBoost': XGBClassifier(tree_method="gpu_hist", use_label_encoder=False, verbosity=0, eval_metric='mlogloss', objective='multi:softmax', num_class=n_classes, seed=seed),
                'LogisticRegression': LogisticRegression(max_iter=10000000),
                'KNN': KNeighborsClassifier(),
                'RandomForest': RandomForestClassifier(),
                'DecisionTree': DecisionTreeClassifier(),
                'LDA': LinearDiscriminantAnalysis(),
                'SVM': SVC(probability=True),
                'LightGBM': LGBMClassifier(device='gpu', objective='multiclass', metric='multi_logloss', num_class=n_classes, verbose=0, seed=seed),
                'CatBoost': CatBoostClassifier(task_type="GPU", eval_metric='MultiClass', classes_count=n_classes, verbose=0, random_seed=seed),
            }

        # Perform k-fold cross-validation
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        print(f'Number of folds: {args.folds}')

        # Perform cross-validation for each algorithm
        for algorithm_name, algorithm in algorithms.items():
            print(f'Running {algorithm_name}...')

            start = time.time()

            if n_classes == 2:
                scoring = {'acc': 'accuracy',
                        'f1_weighted': 'f1_weighted',
                        'roc_auc': 'roc_auc', 
                        'precision': 'precision', 
                        'recall': 'recall',
                        'neg_log_loss': 'neg_log_loss'
                        }
            else:
                scoring = {'acc': 'accuracy',
                            'f1_weighted': 'f1_weighted',
                            'roc_auc': 'roc_auc_ovo',
                            'precision': 'precision_weighted',
                            'recall': 'recall_weighted',
                            'neg_log_loss': 'neg_log_loss'
                            }

            scores = cross_validate(algorithm, X_train, y_train, cv=kf, scoring=scoring, return_train_score=True)

            # Calculate the average scores
            scores_test = {key: value for key, value in scores.items() if key.startswith('test')}
            scores_train = {key: value for key, value in scores.items() if key.startswith('train')}
            avg_scores_test = {key: value.mean() for key, value in scores_test.items()}
            avg_scores_train = {key: value.mean() for key, value in scores_train.items()}

            std_scores_test = {key: value.std() for key, value in scores_test.items()}
            std_scores_train = {key: value.std() for key, value in scores_train.items()}

            end = time.time()

            csv_writer(data_val_path, algorithm_name, avg_scores_train, std_scores_train,
                        avg_scores_test, std_scores_test, end-start)

            algorithm_fit = algorithm.fit(X_train, y_train)

            save_model(algorithm_fit, data_model_path, algorithm_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Disease Classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str, default='train', help='Mode: data_prep, train')
    parser.add_argument('--data', type=str, default='Cardiovascular-Disease-dataset', help='Dataset names')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    main(args)

