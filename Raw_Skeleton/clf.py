# VAE latent regression: same CV as RVAE (5 val + 5 test per fold, two rounds)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from val_test import val_test
from print_results import print_results_clf
import numpy as np
from tqdm.notebook import tqdm
SEED = 42


def clf(zdf, y_lesion, participant_ids):

    n = len(y_lesion)
    n_folds = 30

    models = {'KNN': KNeighborsClassifier()}
    all_results_validation = {name: {'targets': [], 'preds': []} for name in models.keys()}
    all_results_test = {name: {'targets': [], 'preds': [], 'subjects': []} for name in models.keys()}
    participant_ids = np.asarray(participant_ids)


    for k in tqdm(range(n_folds), total=n_folds, desc='VAE folds'):
        validation_pids_list, test_pids_list = val_test(participant_ids, k)
        validation_pids = set(validation_pids_list)
        test_pids = set(test_pids_list)
        train_pids = set(participant_ids) - validation_pids - test_pids

        train_idx = np.array([j for j in range(n) if participant_ids[j] in train_pids])
        validation_idx = np.array([j for j in range(n) if participant_ids[j] in validation_pids])
        test_idx = np.array([j for j in range(n) if participant_ids[j] in test_pids])
        if len(train_idx) == 0 or len(validation_idx) == 0 or len(test_idx) == 0:
            continue

        fold_seed = SEED + k
        Z_train_fold = zdf.iloc[train_idx].loc[:,:'z5'].values
        Z_val_fold = zdf.iloc[validation_idx].loc[:,:'z5'].values
        Z_test_fold = zdf.iloc[test_idx].loc[:,:'z5'].values
        y_train_fold = y_lesion[train_idx]
        y_val_fold = y_lesion[validation_idx]
        y_test_fold = y_lesion[test_idx]

        for name, model_reg in models.items():
            m = type(model_reg)(**model_reg.get_params())
            m.fit(Z_train_fold, y_train_fold)
            validation_preds = m.predict(Z_val_fold)
            test_preds = m.predict(Z_test_fold)
            
            all_results_validation[name]['targets'].extend(y_val_fold.tolist())
            all_results_validation[name]['preds'].extend(validation_preds.tolist())
            all_results_test[name]['targets'].extend(y_test_fold.tolist())
            all_results_test[name]['preds'].extend(test_preds.tolist())
            all_results_test[name]['subjects'].extend(participant_ids[test_idx].tolist())
            
            acc_val = accuracy_score(y_val_fold, validation_preds)
            f1_val = f1_score(y_val_fold, validation_preds, average='macro')
            
            print(f"Fold {k + 1:02d} | {name} | Validation: acc={acc_val:.2f}, F1={f1_val:.2f}")

    results_test_df = print_results_clf(all_results_validation, all_results_test, models)
    return results_test_df