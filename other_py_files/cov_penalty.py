import torch

def cov_penalty(mu, k=38):
    #latent means
    Z = mu                      # (N, R)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    
    Zk = Zc[:, :k]
    Sigma_k = (Zk.T @ Zk) / (Zk.shape[0] - 1)

    diag_k = torch.diag(torch.diag(Sigma_k))
    off_diag_k = Sigma_k - diag_k

    cov_penalty = torch.sum(off_diag_k ** 2)
    return cov_penalty