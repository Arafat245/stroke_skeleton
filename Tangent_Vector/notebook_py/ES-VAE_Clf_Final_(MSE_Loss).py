#!/usr/bin/env python
# coding: utf-8

# ## ES-VAE with MSE Loss (Classification)

# In[1]:


import numpy as np
import pandas as pd
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset

from val_test import *
from print_results import *
from functionsgpu_fast import *

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
dtype = torch.float32

if device.type == "cuda":
    idx = device.index if device.index is not None else torch.cuda.current_device()
    print(torch.cuda.get_device_name(idx))

SEED = 42
def deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

deterministic(SEED)
# Enable (as much as possible) deterministic operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tslen = 200


# ## Load y_lesion and Participant IDs

# In[2]:


# Load y_lesion and participant IDs
# Participant ID for each row of dataset (same order as files from csv_r)
participant_ids = np.loadtxt('/mnt/sdb/arafat/stroke_riemann/labels_data/pids.txt')
demo_df = pd.read_csv('/mnt/sdb/arafat/stroke_riemann/labels_data/demo_data.csv')
id_to_lesion = dict(zip(demo_df['s'].astype(int), demo_df['LesionLeft']))
y_lesion = np.array([id_to_lesion[int(pid)] for pid in participant_ids])

print("y_lesion shape:", y_lesion.shape)
print("First 10 participant_ids:", participant_ids[:10])


# ## Data Loading

# In[3]:


def loading(filename, tslen):
    with open('{}/betas_aligned{}.pkl'.format(filename, tslen), 'rb') as f:
        betas_aligned = pickle.load(f)
    with open('{}/mu{}.pkl'.format(filename, tslen), 'rb') as f:
        mu = pickle.load(f)
    with open('{}/tangent_vecs{}.pkl'.format(filename, tslen), 'rb') as f:
        tangent_vec_all = pickle.load(f)
    return betas_aligned, mu, tangent_vec_all

betas_aligned_all, mu_all_t, tangent_vec_all = loading('/mnt/sdb/arafat/stroke_riemann/aligned_data',tslen)
mu_all_t_tensor = torch.from_numpy(mu_all_t).to(device=device, dtype=torch.float32)
betas_aligned = np.array(betas_aligned_all)
betas_aligned = betas_aligned.transpose(1, 2, 3, 0)
print(betas_aligned.shape, tangent_vec_all.shape, mu_all_t.shape)


# In[4]:


K = 32
M = 3
T = tslen
nsamples = 155

tangent_flat = tangent_vec_all.reshape((K*M*T, nsamples))
print(tangent_flat.shape)


# ## Nonlinear Tangent VAE

# In[5]:


class NonlinearVAE(nn.Module):
    """NonlinearVAE"""
    def __init__(self, D, R, H=128, dropout=0.1):
        super().__init__()
        # Encoder layers
        self.W1 = nn.Linear(D, H, bias=False)        # input -> hidden
        self.W2_mu = nn.Linear(H, R, bias=False)     # hidden -> latent mean
        self.W2_logvar = nn.Linear(H, R)             # hidden -> latent logvar
        self.dropout = nn.Dropout(p=dropout)
        
        # Decoder layers
        self.dec1 = nn.Linear(R, 32, bias=False)
        self.dec2 = nn.Linear(32, D, bias=False)

    def encode(self, x):
        h = torch.tanh(self.W1(x))
        h = self.dropout(h)
        mu = self.W2_mu(h)
        logvar = self.W2_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h_recon = torch.tanh(self.dec1(z))
        x_hat = self.dec2(h_recon)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

def vae_loss(x, x_hat, mu, logvar, beta=1e-4):
    dist = (x-x_hat)**2
    recon = torch.mean(dist.sum(dim=1))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    avg_kl = kl.mean()
    return recon + beta * avg_kl, recon, avg_kl


# ## Training Function for Each Fold (VAE)

# In[6]:


def train_vae_fold(X_tan_train, D, R, num_epochs=1000, lr=1e-3, betakl=2**(-3), 
                   batch_size=32, verbose=False, seed=42):
    """Train a fresh NonlinearVAE on a training subset only (tangent space MSE + KL loss).

    X_tan_train : (N_train, D) tangent vectors
    """
    deterministic(seed)

    dataset = TensorDataset(X_tan_train)
    g = torch.Generator(device=device).manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    model_fold = NonlinearVAE(D, R).to(device=device, dtype=dtype)
    opt_fold = torch.optim.Adam(model_fold.parameters(), lr=lr)

    model_fold.train()
    for epoch in range(num_epochs):
        epoch_loss, epoch_recon, epoch_kl, num_samples = 0.0, 0.0, 0.0, 0
        
        for (x_batch,) in loader:
            x_batch = x_batch.to(device=device, dtype=dtype)
            opt_fold.zero_grad(set_to_none=True)
            x_hat, mu, logvar, z = model_fold(x_batch)
            loss_train, recon_train, kl_train = vae_loss(x_batch, x_hat, mu, logvar, beta=betakl)
            loss_train.backward()
            opt_fold.step()

            bs = x_batch.size(0)
            epoch_loss += loss_train.item() * bs
            epoch_recon += recon_train.item() * bs
            epoch_kl += kl_train.item() * bs
            num_samples += bs

        if verbose and (epoch % 300 == 0 or epoch == num_epochs - 1):
            avg_loss = epoch_loss / num_samples
            avg_recon = epoch_recon / num_samples
            avg_kl = epoch_kl / num_samples
            print(f"[fold VAE] epoch {epoch} | loss {avg_loss:.6f} | recon {avg_recon:.6f} | kl {avg_kl:.6f}")
            
    model_fold.eval()
    return model_fold


# ## VAE Cross Validation

# In[7]:


# VAE latent regression: same CV as RVAE (5 val + 5 test per fold, two rounds)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

dtype = torch.float32
n = len(y_lesion)
D = tangent_flat.shape[0]
n_folds = 30
R_vae = 38

models = {'KNN': KNeighborsClassifier()}
all_results_validation_vae = {name: {'targets': [], 'preds': []} for name in models.keys()}
all_results_test_vae = {name: {'targets': [], 'preds': [], 'subjects': []} for name in models.keys()}
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
    X_tan_train = torch.from_numpy(tangent_flat[:, train_idx].T.astype(np.float32)).to(device=device, dtype=dtype)
    
    bsize = X_tan_train.shape[0]
    
    model_fold = train_vae_fold(X_tan_train, D, R_vae, num_epochs=50, lr=1e-3, 
                                batch_size=bsize, seed=fold_seed)

    with torch.no_grad():
        mu_train_fold, _ = model_fold.encode(X_tan_train)
        mu_val_fold, _ = model_fold.encode(torch.from_numpy(tangent_flat[:, validation_idx].T.astype(np.float32)).to(device=device, dtype=dtype))
        mu_test_fold, _ = model_fold.encode(torch.from_numpy(tangent_flat[:, test_idx].T.astype(np.float32)).to(device=device, dtype=dtype))

    Z_train_fold = mu_train_fold.cpu().numpy()
    Z_val_fold = mu_val_fold.cpu().numpy()
    Z_test_fold = mu_test_fold.cpu().numpy()
    
    y_train_fold = y_lesion[train_idx]
    y_val_fold = y_lesion[validation_idx]
    y_test_fold = y_lesion[test_idx]

    for name, model_reg in models.items():
        m = type(model_reg)(**model_reg.get_params())
        m.fit(Z_train_fold, y_train_fold)
        validation_preds = m.predict(Z_val_fold)
        test_preds = m.predict(Z_test_fold)
        
        all_results_validation_vae[name]['targets'].extend(y_val_fold.tolist())
        all_results_validation_vae[name]['preds'].extend(validation_preds.tolist())
        all_results_test_vae[name]['targets'].extend(y_test_fold.tolist())
        all_results_test_vae[name]['preds'].extend(test_preds.tolist())
        all_results_test_vae[name]['subjects'].extend(participant_ids[test_idx].tolist())
        
        acc_val = accuracy_score(y_val_fold, validation_preds)
        f1_val = f1_score(y_val_fold, validation_preds, average='macro')
        
        print(f"Fold {k + 1:02d} | {name} | Validation: acc={acc_val:.2f}, F1={f1_val:.2f}")

results_test_vae_df = print_results_clf(all_results_validation_vae, all_results_test_vae, models)
results_test_vae_df


# In[8]:


from ci_class import *

ci_results = {}

name = "KNN"

ci_results[name] = subject_bootstrap_ci_class(
    all_results_test_vae[name]['targets'],
    all_results_test_vae[name]['preds'],
    all_results_test_vae[name]['subjects'])

pd.DataFrame(ci_results['KNN'])


# In[ ]:




