import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
import pandas as pd

def print_results_regression(all_results_validation, all_results_test, models):

    results_validation = {}
    results_test = {}

    for name in models.keys():
        t = np.array(all_results_validation[name]['targets'])
        p = np.array(all_results_validation[name]['preds'])
        results_validation[name] = {
            'MAE': mean_absolute_error(t, p), 'RMSE': np.sqrt(mean_squared_error(t, p)), 'R2': r2_score(t, p),
            'Pearson r': stats.pearsonr(t, p)[0], 'Pearson p': stats.pearsonr(t, p)[1],
        }

        t = np.array(all_results_test[name]['targets'])
        p = np.array(all_results_test[name]['preds'])
        results_test[name] = {
            'MAE': mean_absolute_error(t, p), 'RMSE': np.sqrt(mean_squared_error(t, p)), 'R2': r2_score(t, p),
            'Pearson r': stats.pearsonr(t, p)[0], 'Pearson p': stats.pearsonr(t, p)[1],
        }

    print("\n=== Validation Performance (across all folds) ===")
    results_validation_df = pd.DataFrame(results_validation).T
    print(results_validation_df)

    print("\n=== Test Performance (across all folds) ===")
    results_test_df = pd.DataFrame(results_test).T
    return results_test_df


def print_results_clf(all_results_validation, all_results_test, models):

    results_validation = {}
    results_test = {}

    for name in models.keys():
        t = np.array(all_results_validation[name]['targets'])
        p = np.array(all_results_validation[name]['preds'])
        results_validation[name] = {
            'Accuracy': accuracy_score(t, p), 'F1 (macro)': f1_score(t, p, average='macro'), 
            'Precision (macro)': precision_score(t, p, average='macro'), 'Recall (macro)': recall_score(t, p, average='macro'),
        }

        t = np.array(all_results_test[name]['targets'])
        p = np.array(all_results_test[name]['preds'])
        results_test[name] = {
            'Accuracy': accuracy_score(t, p), 'F1 (macro)': f1_score(t, p, average='macro'), 
            'Precision (macro)': precision_score(t, p, average='macro'), 'Recall (macro)': recall_score(t, p, average='macro'),
        }

    print("\n=== Validation Performance (across all folds) ===")
    results_validation_df = pd.DataFrame(results_validation).T
    print(results_validation_df)

    print("\n=== Test Performance (across all folds) ===")
    results_test_df = pd.DataFrame(results_test).T
    return results_test_df