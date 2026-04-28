import torch

def corr_penalty(mu, tau=0.6):
    Z = mu
    Zc = Z - Z.mean(dim=0, keepdim=True)
    
    Sigma = (Zc.T @ Zc) / (Zc.shape[0] - 1)
    
    std = torch.sqrt(torch.diag(Sigma) + 1e-8)
    D_inv = torch.diag(1.0 / std)
    Rho = D_inv @ Sigma @ D_inv
    
    off = Rho - torch.diag(torch.diag(Rho))
    
    hinge = torch.relu(off.abs() - tau)
    corr_penalty = torch.sum(hinge ** 2)
    return corr_penalty