#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import random
import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

from data_utils_load import *
from tcn import TCN

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def filter_valid_files(file_list):
    valid_files = []
    for file in file_list:
        file = file.strip()
        if re.match(r'^ID\d+_\d+\.csv$', file):
            valid_files.append(file)
    return valid_files

def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

bsize = 64


# ## Loading POMA Data

# In[2]:


save_dir = 'data'
loaded_dir = 'csv_r'
# preprocess_and_save_data(loaded_dir, save_dir)
all_files = filter_valid_files(os.listdir(loaded_dir))
participant_ids = sorted({re.search(r'ID(\d+)_', f).group(1) for f in all_files}, key=lambda x: int(x))


# ## Gait Unit Creation

# In[2]:


class GaitUnitsDataset(Dataset):
    def __init__(self, x_load_list, y_batch, N_gaits):
        self.unit_samples = []
        self.unit_labels = []
        
        for x_task, y_task in zip(x_load_list, y_batch):
            L = x_task.size(0)
            num_units = L - N_gaits + 1
            units = [x_task[start:start + N_gaits] for start in range(num_units)]
            self.unit_samples.extend(units)
            self.unit_labels.extend([y_task] * num_units)
        print(f"Created {len(self.unit_samples)} gait windows")
    
    def __len__(self):
        return len(self.unit_samples)
    
    def __getitem__(self, idx):
        return self.unit_samples[idx], self.unit_labels[idx]


def create_gait_units_dataloader(x_load_list, y_batch, N_gaits, batch_size=64, shuffle=True):
    dataset = GaitUnitsDataset(x_load_list, y_batch, N_gaits)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ## Regression Evaluation Function

# In[3]:


def evaluate_model(model, test_loader, n_select_gaits):

    model.eval()
    device = next(model.parameters()).device

    t = []
    p = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x_load_list, y_batch, _ = batch
            else:
                x_load_list, y_batch = batch

            for i, x_load_sample in enumerate(x_load_list):
                x_task = x_load_sample[:6]
                num_units = len(x_task) - n_select_gaits + 1

                unit_preds = []
                    
                for start_idx in range(num_units):
                    unit = x_task[start_idx:start_idx + n_select_gaits]
                    unit_tensor = unit.unsqueeze(0).to(device)
                    y_target = y_batch[i:i+1].to(device)
                        
                    y_pred = model(unit_tensor)
                    unit_preds.append(y_pred.item())

                participant_target = y_target.item()
                participant_pred = float(np.median(np.array(unit_preds)))

                t.append(participant_target)
                p.append(participant_pred)
    
    return t, p


# ## Regression Model Training Function

# In[4]:


def train_model(model, train_loader, experiment_seed=42, num_epochs=1000, N_gaits=2, val_loader=None, patience=50):
    set_deterministic(experiment_seed)
    random.seed(experiment_seed)
    np.random.seed(experiment_seed)

    all_x = []
    all_y = []
    
    for batch in train_loader:
        if len(batch) == 3:
            x_load_list, y_batch, _ = batch
        else:
            x_load_list, y_batch = batch
        all_x.extend(x_load_list)
        all_y.extend(y_batch)
    
    optimized_train_loader = create_gait_units_dataloader(all_x, all_y, N_gaits=N_gaits, 
                                                          batch_size=bsize, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_mae = float('inf')
    best_state = None
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in optimized_train_loader:
            x_batch = x_batch.to(device) 
            y_batch = y_batch.to(device)  
            
            y_pred = model(x_batch).view(-1)
            loss = F.l1_loss(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()

        if val_loader is not None:
            val_t, val_p = evaluate_model(model, val_loader, n_select_gaits=N_gaits)
            val_mae = np.mean(np.abs(np.array(val_t) - np.array(val_p)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(optimized_train_loader)
            msg = f"[Epoch {epoch + 1}/{num_epochs}] Train MAE: {avg_loss:.4f}"
            if val_loader is not None:
                msg += f" | Val MAE: {val_mae:.4f}"
            print(msg)

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model


# ## TCN Regressor Training

# In[6]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from val_test import val_test
from print_results import print_results

n_folds = 30
models = {'TCN': []}
name = list(models.keys())[0]
all_results_validation = {name: {'targets': [], 'preds': []}}
all_results_test = {name: {'targets': [], 'preds': [], 'subjects': []}}


for k in tqdm(range(n_folds)):
    
    model = TCN(N_gaits=2).to(device)

    validation_pids, test_pids = val_test(participant_ids, k)
    print(f"Fold {k+1} | Validation Subs: {validation_pids} | Test Subs: {test_pids}")

    train_loader, val_loader, test_loader = cv_data_create(validation_pids, test_pids, save_dir, batch_size=bsize)
    trained_model = train_model(model, train_loader, experiment_seed=k, val_loader=val_loader, patience=200)

    val_t, val_p = evaluate_model(trained_model, val_loader, n_select_gaits=2)
    test_t, test_p = evaluate_model(trained_model, test_loader, n_select_gaits=2)

    all_results_validation[name]['targets'].extend(val_t)
    all_results_validation[name]['preds'].extend(val_p)
    all_results_test[name]['targets'].extend(test_t)
    all_results_test[name]['preds'].extend(test_p)
    all_results_test[name]['subjects'].extend(test_pids)

    mae_val = mean_absolute_error(val_t, val_p)
    rmse_val = np.sqrt(mean_squared_error(val_t, val_p))
    r2_val = r2_score(val_t, val_p)
    print(f"Fold {k + 1:02d} | {name} | Validation: MAE={mae_val:.3f}, RMSE={rmse_val:.3f}, R2={r2_val:.3f}\n")

test_results_df = print_results(all_results_validation, all_results_test, models)
test_results_df


# In[8]:


from ci import *

ci_results = {}

name = "TCN"

ci_results[name] = subject_bootstrap_ci(
    all_results_test[name]['targets'],
    all_results_test[name]['preds'],
    all_results_test[name]['subjects'])

pd.DataFrame(ci_results['TCN'])


# ## Loading Lesion Left Classifcation Data

# In[6]:


save_dir = 'data_clf'
loaded_dir = 'csv_clf'
preprocess_and_save_data(loaded_dir, save_dir)
all_files = filter_valid_files(os.listdir(loaded_dir))
participant_ids = sorted({re.search(r'ID(\d+)_', f).group(1) for f in all_files}, key=lambda x: int(x))


# ## Classification Evaluation Function

# In[9]:


def evaluate_model_clf(model, test_loader, n_select_gaits, num_classes=3):
    """Classification evaluation: aggregate unit-level predictions per participant (majority vote)."""
    model.eval()
    device = next(model.parameters()).device

    t = []
    p = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x_load_list, y_batch, _ = batch
            else:
                x_load_list, y_batch = batch

            for i, x_load_sample in enumerate(x_load_list):
                x_task = x_load_sample[:6]
                num_units = len(x_task) - n_select_gaits + 1

                unit_preds = []

                for start_idx in range(num_units):
                    unit = x_task[start_idx:start_idx + n_select_gaits]
                    unit_tensor = unit.unsqueeze(0).to(device)

                    logits = model(unit_tensor)
                    pred_class = logits.argmax(dim=1).item()
                    unit_preds.append(pred_class)

                participant_target = y_batch[i].item()
                participant_pred = int(np.bincount(unit_preds).argmax())

                t.append(participant_target)
                p.append(participant_pred)

    return t, p


# ## Classification Model Training Function

# In[11]:


def train_model_clf(model, train_loader, experiment_seed=42, num_epochs=1000, N_gaits=2, val_loader=None, patience=50, num_classes=3):
    set_deterministic(experiment_seed)
    random.seed(experiment_seed)
    np.random.seed(experiment_seed)

    all_x = []
    all_y = []

    for batch in train_loader:
        if len(batch) == 3:
            x_load_list, y_batch, _ = batch
        else:
            x_load_list, y_batch = batch
        all_x.extend(x_load_list)
        all_y.extend(y_batch)

    optimized_train_loader = create_gait_units_dataloader(all_x, all_y, N_gaits=N_gaits,
                                                          batch_size=bsize, shuffle=True)

    val_unit_loader = None
    if val_loader is not None:
        val_x, val_y = [], []
        for batch in val_loader:
            if len(batch) == 3:
                x_load_list, y_batch, _ = batch
            else:
                x_load_list, y_batch = batch
            val_x.extend(x_load_list)
            val_y.extend(y_batch)
        val_unit_loader = create_gait_units_dataloader(val_x, val_y, N_gaits=N_gaits,
                                                       batch_size=bsize, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for x_batch, y_batch in optimized_train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).long()

            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if val_unit_loader is not None:
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for x_batch, y_batch in val_unit_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device).long()
                    logits = model(x_batch)
                    val_loss_sum += F.cross_entropy(logits, y_batch, reduction='sum').item()
                    val_n += x_batch.size(0)
            val_loss = val_loss_sum / val_n
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(optimized_train_loader)
            msg = f"[Epoch {epoch + 1}/{num_epochs}] Train CE loss: {avg_loss:.4f}"
            if val_unit_loader is not None:
                val_t, val_p = evaluate_model_clf(model, val_loader, n_select_gaits=N_gaits, num_classes=num_classes)
                val_acc = np.mean(np.array(val_t) == np.array(val_p))
                msg += f" | Val CE loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            print(msg)

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model


# ## TCN Classification Training

# In[17]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from val_test import val_test

n_folds = 30
models_clf = {'TCN': []}
name = list(models_clf.keys())[0]

all_results_validation_clf = {name: {'targets': [], 'preds': []}}
all_results_test_clf = {name: {'targets': [], 'preds': [], 'subjects': []}}

for k in tqdm(range(n_folds)):

    model = TCN(N_gaits=2, output_dim=3).to(device)  # 3-class logits

    validation_pids, test_pids = val_test(participant_ids, k)
    print(f"Fold {k+1} | Validation Subs: {validation_pids} | Test Subs: {test_pids}")

    train_loader, val_loader, test_loader = cv_data_create(validation_pids, test_pids, save_dir, batch_size=bsize)
    trained_model = train_model_clf(model, train_loader, experiment_seed=k, num_epochs=1000, N_gaits=2,
                                    val_loader=val_loader, patience=150, num_classes=3)

    val_t, val_p = evaluate_model_clf(trained_model, val_loader, n_select_gaits=2, num_classes=3)
    test_t, test_p = evaluate_model_clf(trained_model, test_loader, n_select_gaits=2, num_classes=3)

    all_results_validation_clf[name]['targets'].extend(val_t)
    all_results_validation_clf[name]['preds'].extend(val_p)

    all_results_test_clf[name]['targets'].extend(test_t)
    all_results_test_clf[name]['preds'].extend(test_p)
    all_results_test_clf[name]['subjects'].extend(test_pids)

    acc_val = accuracy_score(val_t, val_p)
    f1m_val = f1_score(val_t, val_p, average='macro')
    print(f"Fold {k + 1:02d} | {name} | Validation: Acc={acc_val:.3f}, F1-macro={f1m_val:.3f}\n")

# Overall pooled (OOF) metrics across all folds
val_t_all = np.array(all_results_validation_clf[name]['targets'])
val_p_all = np.array(all_results_validation_clf[name]['preds'])
test_t_all = np.array(all_results_test_clf[name]['targets'])
test_p_all = np.array(all_results_test_clf[name]['preds'])

def classification_metrics(t, p):
    return {
        'Accuracy': accuracy_score(t, p),
        'F1 (weighted)': f1_score(t, p, average='weighted'),
        'F1 (macro)': f1_score(t, p, average='macro'),
        'Precision (weighted)': precision_score(t, p, average='weighted', zero_division=0),
        'Precision (macro)': precision_score(t, p, average='macro', zero_division=0),
        'Recall (weighted)': recall_score(t, p, average='weighted', zero_division=0),
        'Recall (macro)': recall_score(t, p, average='macro', zero_division=0),
    }

results_val = classification_metrics(val_t_all, val_p_all)
results_test = classification_metrics(test_t_all, test_p_all)

print("\n=== Validation Performance (across all folds) ===")
print(pd.DataFrame({name: results_val}).T)

print("\n=== Test Performance (across all folds) ===")
results_test_df = pd.DataFrame({name: results_test}).T
results_test_df


# In[18]:


from ci_class import subject_bootstrap_ci_class

ci_results = {}

name = "TCN"

ci_results[name] = subject_bootstrap_ci_class(
    all_results_test_clf[name]['targets'],
    all_results_test_clf[name]['preds'],
    all_results_test_clf[name]['subjects'])

pd.DataFrame(ci_results['TCN'])

