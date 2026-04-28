import os
import warnings
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import torch
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda:1')
device = torch.device('cuda:1')

import geomstats.backend as gs
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.visualization import KendallDisk, KendallSphere
import fdasrsf
import matplotlib.pyplot as plt
from geomstats.geometry.matrices import Matrices
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

preshape = PreShapeSpace(32, 3)
# preshape = PreShapeSpace(3, 2)
preshape.equip_with_group_action("rotations")
preshape.equip_with_quotient()


# def OPA_gpu(A_t, B_t, reflect=False):
    
#     """ Alignes A to B """       
#     if reflect:
#         # Correctly transpose A_t and B_t to enable proper matrix multiplication
#         A = A_t.permute(2, 1, 0)  # (200, 3, 29)
#         B = B_t.permute(2, 1, 0)  # (200, 3, 29)
        
#         # Transpose A to get (200, 29, 3)
#         A_transposed = A.permute(0, 2, 1)  # (200, 29, 3)

#         # Now perform batch matrix multiplication across the 'sample' dimension
#         product = B @ A_transposed
#         u, sigma, v_t = torch.linalg.svd(product, full_matrices=False)
#         R = u @ v_t

#         # Apply R to the original A matrix, properly transposed to match dimensions
#         aligned = R @ A
        
#         # returning numpy array
#         return aligned.permute(2, 1, 0)  # Back to (29, 3, 200)
#     else:
#         # converting (29, 3, 200) to (200, 29, 3)
#         A = A_t.permute(2, 0, 1)
#         B = B_t.permute(2, 0, 1)
        
#         # aligning 
#         aligned =  Matrices.align_matrices(A, B)
#         # returning numpy array
#         return aligned.permute(1, 2, 0)  # Back to (29, 3, 200)


# def rotate_trajectory_align_gpu(mu, traj, reflect=False):
#     """Batch-aligns trajectories with mu using the modified OPA."""
#     # Call the batched OPA directly
#     traj_aligned = OPA_gpu(traj, mu, reflect=reflect)
#     return traj_aligned


# ----- Batched GPU alignment (fast path) -----

def OPA_gpu_batch(mu, betas):
    """Batched OPA: align N trajectories (N, 29, 3, 200) to mu (29, 3, 200). Returns (N, 29, 3, 200)."""
    N = betas.shape[0]
    # (N, 29, 3, 200) -> (N, 200, 29, 3)
    A = betas.permute(0, 3, 1, 2)
    B = mu.permute(2, 0, 1).unsqueeze(0).expand(N, -1, -1, -1)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    aligned = Matrices.align_matrices(A.double(), B.double())
    return aligned.permute(0, 2, 3, 1).float()  # (N, 29, 3, 200)


def rotate_trajectory_align_batch_gpu(mu, betas):
    """Batched rotation alignment: betas (N, 29, 3, 200) -> (N, 29, 3, 200)."""
    return OPA_gpu_batch(mu, betas)


def log_gpu(p1, p2):
    # p1: landmarks x ambient
    # p2: landmarks x ambient
    p1 = p1.permute(2, 0, 1)  # (29, 3, 200) to (200, 29, 3)
    p2 = p2.permute(2, 0, 1)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    result = preshape.quotient.metric.log(p2.double(), p1.double())
    return result.permute(1, 2, 0).float()
    # return preshape.metric.log(p1, p2)

def log_gpu_frechet(p1, p2):
    # p1: (29, 3, 200), a single set of landmarks across time
    # p2: (130, 29, 3, 200), a batch of sets of landmarks across time
    
    # First, add a new dimension at the beginning of p1, then expand to match p2's size
    p1_expanded = p1.unsqueeze(0).expand(p2.shape[0], -1, -1, -1)  # Expand p1 to (130, 29, 3, 200)
    
    # Permute to match the expected dimensions for processing
    p1_expanded = p1_expanded.permute(0, 3, 1, 2)  # (130, 200, 29, 3)
    p2 = p2.permute(0, 3, 1, 2)  # (130, 200, 29, 3)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    result = preshape.quotient.metric.log(p2.double(), p1_expanded.double())
    # Permute back to the original dimensions
    result = result.permute(2, 3, 1, 0).float()  # Back to (29, 3, 200, 130)

    return result


def exp_gpu(p, v):
    # p: landmarks x ambient
    p = p.permute(2, 0, 1)
    v = v.permute(2, 0, 1)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    result = preshape.metric.exp(v.double(), p.double())
    return result.permute(1, 2, 0).float()


def log_gpu_batch(p1, p2):
    """Batched log map: p1, p2 (N, 29, 3, 200) -> (N, 29, 3, 200)."""
    p1 = p1.permute(0, 3, 1, 2)  # (N, 200, 29, 3)
    p2 = p2.permute(0, 3, 1, 2)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    p1_d = p1.double()
    p2_d = p2.double()
    result = preshape.quotient.metric.log(p2_d, p1_d)
    return result.permute(0, 2, 3, 1).float()


def exp_gpu_batch(p, v):
    """Batched exp map: p, v (N, 29, 3, 200) -> (N, 29, 3, 200)."""
    p = p.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    p_d = p.double()
    v_d = v.double()
    result = preshape.metric.exp(v_d, p_d)
    return result.permute(0, 2, 3, 1).float()


def parallel_gpu(v, p1, p2, n_steps=10):
    v = v.permute(2, 0, 1)  # (29, 3, 200) to (200, 29, 3)
    p1 = p1.permute(2, 0, 1)
    p2 = p2.permute(2, 0, 1)
    # geomstats uses float64 internally; cast to avoid "Float did not match Double"
    result = preshape.quotient.metric.parallel_transport(tangent_vec=v.double(), base_point=p1.double(), end_point=p2.double(), n_steps=2)
    return result.permute(1, 2, 0).float()
    

def preprocess(x):
    """Removes translations and scaling from a k (landmarks) x m (ambient dimension, eg. 2 for 2d shapes)"""
    mu = x.mean(axis=0)
    for i in range(x.shape[0]):
        x[i, :] = x[i, :] - mu
    x = x / np.linalg.norm(x, ord="fro")
    return x
    

def preprocess_temporal(data):
    """Mean centers data and removes scaling for kendall shape space"""
    for t in range(data.shape[2]):
        data[:, :, t] = preprocess(data[:, :, t])

    return data


def cov_der_gpu(beta_t, delta_t, c):
    """Given a function, calculate a derivative in the tangent space of beta(t)"""
    beta_dot_t = torch.zeros_like(beta_t)

    # Compute the log differences for all t except the last one, in parallel
    beta_dot_t[:, :, :-1] = log_gpu(beta_t[:, :, :-1], beta_t[:, :, 1:]) / delta_t

    # Handle the last point separately if required
    pt = parallel_gpu(beta_dot_t[:, :, -2].unsqueeze(2), beta_t[:, :, -2].unsqueeze(2), beta_t[:, :, -1].unsqueeze(2))
    beta_dot_t[:, :, -1] = pt.squeeze(2)

    # Convert the result back to a numpy array
    return beta_dot_t


def parallel_vf_gpu(v, beta, c):
    """Transports the entire vector field v to the tangent spaces at beta to the tangent space of reference point c"""
    c = c.unsqueeze(2)
    
    # Use parallel_gpu to process the entire batch
    parallel_vf = parallel_gpu(v, beta, c)
    
    return parallel_vf


def cov_der_gpu_batch(beta_t, delta_t, c):
    """Batched cov derivative: beta_t (N, 29, 3, 200) -> (N, 29, 3, 200)."""
    N = beta_t.shape[0]
    beta_dot_t = torch.zeros_like(beta_t)
    beta_dot_t[:, :, :, :-1] = log_gpu_batch(
        beta_t[:, :, :, :-1], beta_t[:, :, :, 1:]
    ) / delta_t
    for n in range(N):
        pt = parallel_gpu(
            beta_dot_t[n, :, :, -2].unsqueeze(2),
            beta_t[n, :, :, -2].unsqueeze(2),
            beta_t[n, :, :, -1].unsqueeze(2),
        )
        beta_dot_t[n, :, :, -1] = pt.squeeze(2)
    return beta_dot_t


def parallel_vf_gpu_batch(v, beta, c):
    """Batched parallel transport: v, beta (N, 29, 3, 200), c (29, 3) -> (N, 29, 3, 200)."""
    N = v.shape[0]
    c_exp = c.unsqueeze(2)  # (29, 3, 1)
    out = torch.zeros_like(v)
    for n in range(N):
        out[n] = parallel_gpu(v[n], beta[n], c_exp)
    return out


def srvf_gpu(beta_dot_t, delta_t):
    """Single (29, 3, T) or batch (N, 29, 3, T)."""
    # Norm over landmarks and ambient: single -> (1,1,T), batch -> (N,1,1,T)
    dims = (1, 2) if beta_dot_t.dim() == 4 else (0, 1)
    norms = torch.linalg.norm(beta_dot_t, dim=dims, keepdim=True)
    thresh = 0.0000001
    norms = torch.clamp(norms, min=thresh)
    return beta_dot_t / norms


def tsrvf(beta_t, delta_t, c):
    # beta_dot = cov_der(beta_t, delta_t)

    # beta_parallel = parallel_vf(beta_dot, beta, c)
    beta_dot = cov_der_gpu(beta_t, delta_t, c)

    beta_dot_c = parallel_vf_gpu(beta_dot, beta_t, c)

    q_t = srvf_gpu(beta_dot_c, delta_t)

    return q_t


def tsrvf_batch(mu, betas, delta_t):
    """Batched TSRVF: mu (29, 3, 200), betas (N, 29, 3, 200). Returns q_mu (29, 3, 200), q_betas (N, 29, 3, 200)."""
    c = mu[:, :, 0]
    q_mu = tsrvf(mu, delta_t, c)
    beta_dot = cov_der_gpu_batch(betas, delta_t, c)
    beta_dot_c = parallel_vf_gpu_batch(beta_dot, betas, c)
    q_betas = srvf_gpu(beta_dot_c, delta_t)
    return q_mu, q_betas


def compose_gpu(beta, t, gamma):
    # Convert numpy arrays to torch tensors and move to GPU
    gamma = torch.from_numpy(gamma).to(device)

    # Pre-compute indices i for each gamma using searchsorted
    i = torch.searchsorted(t, gamma, right=True)
    i = torch.clamp(i, min=1, max=t.shape[0] - 1)  # Ensure indices are valid

    # Calculate delta_t for all gamma simultaneously
    delta_t = (gamma - t[i - 1]) / (t[i] - t[i - 1])

    # Prepare batch operations
    p1 = beta[:, :, i - 1]  # Points from beta corresponding to i-1
    p2 = beta[:, :, i]      # Points from beta corresponding to i

    # Batch compute the log map of the paths
    log_v = log_gpu(p1, p2)  # Perform batch log map
    v = log_v * delta_t.unsqueeze(0).unsqueeze(0)  # Apply delta_t scaling

    # Batch compute the exponential map
    beta_gamma = exp_gpu(p1, v)  # Perform batch exponential map

    return beta_gamma


def compose_batch_gpu(betas, t, gammas):
    """Batched compose: betas (N, 29, 3, 200), t 1D tensor, gammas (N, T) tensor or numpy. Returns (N, 29, 3, 200)."""
    if not isinstance(gammas, torch.Tensor):
        gammas = torch.from_numpy(np.asarray(gammas, dtype=np.float64)).float().to(betas.device)
    elif gammas.device != betas.device:
        gammas = gammas.to(betas.device)
    N, _, _, T = betas.shape
    i = torch.searchsorted(t, gammas, right=True)
    i = torch.clamp(i, min=1, max=t.shape[0] - 1)
    delta_t = (gammas - t[i - 1]) / (t[i] - t[i - 1] + 1e-10)
    ind_prev = (i - 1).unsqueeze(1).unsqueeze(2).expand(-1, betas.shape[1], betas.shape[2], -1).long()
    ind_curr = i.unsqueeze(1).unsqueeze(2).expand(-1, betas.shape[1], betas.shape[2], -1).long()
    p1 = torch.gather(betas, 3, ind_prev)
    p2 = torch.gather(betas, 3, ind_curr)
    log_v = log_gpu_batch(p1, p2)
    v = log_v * delta_t.unsqueeze(1).unsqueeze(2)
    beta_gamma = exp_gpu_batch(p1, v)
    return beta_gamma


def temporal_align(mu, beta, delta_t, method="DP"):
    """Single trajectory: mu, beta (29, 3, 200). Returns gamma_inv (200,) numpy.
    method: 'DP' (default) or 'RBFGS' for fdasrsf optimum_reparam_curve."""
    c = mu[:, :, 0]
    q_mu = tsrvf(mu, delta_t, c)
    q_beta = tsrvf(beta, delta_t, c)
    q_mu_flat = q_mu.reshape(-1, q_mu.shape[2])
    q_beta_flat = q_beta.reshape(-1, q_mu.shape[2])
    with warnings.catch_warnings():
        if method == "RBFGS":
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="fdasrsf")
        # fdasrsf Cython code expects float64 (double) buffers
        q_mu_np = q_mu_flat.cpu().numpy().astype(np.float64)
        q_beta_np = q_beta_flat.cpu().numpy().astype(np.float64)
        gamma_inv = fdasrsf.curve_functions.optimum_reparam_curve(
            q_mu_np, q_beta_np, method=method
        )
    return gamma_inv


def _optimum_reparam_single(q_mu_flat_np, q_beta_flat_n, method="DP"):
    """Helper for parallel reparam: single (D, T) vs (D, T). Returns gamma (T,)."""
    with warnings.catch_warnings():
        if method == "RBFGS":
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="fdasrsf")
        return fdasrsf.curve_functions.optimum_reparam_curve(
            q_mu_flat_np, q_beta_flat_n, method=method
        )


def temporal_align_batch(mu, betas, delta_t, n_jobs_dp=-1, method="DP"):
    """Batched temporal alignment with minimal GPU→CPU sync.
    mu (29, 3, 200), betas (N, 29, 3, 200). Returns list of N gamma_inv arrays (200,).
    method: 'DP' (default) or 'RBFGS'. Single batched tsrvf, one transfer, then N fdasrsf calls (parallel if n_jobs_dp != 0)."""
    q_mu, q_betas = tsrvf_batch(mu, betas, delta_t)
    N, _, _, T = q_betas.shape
    q_mu_flat = q_mu.reshape(-1, T)
    q_betas_flat = q_betas.reshape(N, -1, T)
    # fdasrsf Cython code expects float64 (double) buffers
    q_mu_np = q_mu_flat.cpu().numpy().astype(np.float64)
    q_betas_np = q_betas_flat.cpu().numpy().astype(np.float64)
    if n_jobs_dp != 0:
        gamma_inv_list = Parallel(n_jobs=n_jobs_dp)(
            delayed(_optimum_reparam_single)(q_mu_np, q_betas_np[n], method=method)
            for n in range(N)
        )
    else:
        gamma_inv_list = [
            _optimum_reparam_single(q_mu_np, q_betas_np[n], method=method)
            for n in range(N)
        ]
    return gamma_inv_list


# ----- Old per-trajectory alignment (commented; use batched versions below for speed) -----
# def temporal_rotation_align(mu, beta, t, iterations=10, tol=10 ** (-5), reflect=False):
#     prev_error = -10000
#     delta_t = t[1] - t[0]
#     history = []
#     beta_hat = beta
#     for iteration in range(iterations):
#         error = torch.norm(mu - beta_hat)
#         error = error.item()
#         history.append(error)
#         beta_hat = rotate_trajectory_align_gpu(mu, beta_hat, reflect=reflect)
#         gamma_inv = temporal_align(mu, beta_hat, delta_t)
#         beta_hat = compose_gpu(beta_hat, t, gamma_inv)
#         if abs(error - prev_error) < tol:
#             break
#         else:
#             prev_error = error
#     return beta_hat, gamma_inv, history
#
# def parallel_align(mu, betas, t):
#     N = len(betas)
#     def align(n):
#         return temporal_rotation_align(mu, betas[n], t)
#     results = Parallel(n_jobs=-1)(delayed(align)(n) for n in range(N))
#     betas_aligned, gammas, temp_histories = zip(*results)
#     return list(betas_aligned), list(gammas), list(temp_histories)


def temporal_rotation_align_batch(mu, betas, t, iterations=10, tol=10 ** (-5), n_jobs_temporal=-1, method="DP"):
    """Batched temporal+rotation alignment on GPU; temporal reparam (fdasrsf) on CPU.
    betas: (N, 29, 3, 200). Returns (N, 29, 3, 200), list of N gammas, history list.
    method: 'DP' or 'RBFGS'. 
    For speed you can try fewer iterations, looser tol, or method='RBFGS' if acceptable."""
    delta_t = (t[1] - t[0]).item() if hasattr(t[1] - t[0], 'item') else float(t[1] - t[0])
    prev_error = -10000
    history = []
    beta_hat = betas

    for iteration in range(iterations):
        beta_hat = rotate_trajectory_align_batch_gpu(mu, beta_hat)
        gamma_inv_list = temporal_align_batch(
            mu, beta_hat, delta_t, n_jobs_dp=n_jobs_temporal, method=method
        )
        gammas_stack = np.stack(gamma_inv_list, axis=0)
        gammas_t = torch.from_numpy(gammas_stack).float().to(beta_hat.device)
        beta_hat = compose_batch_gpu(beta_hat, t, gammas_t)
        error = torch.norm(mu.unsqueeze(0) - beta_hat).item()
        history.append(error)
        if abs(error - prev_error) < tol:
            break
        prev_error = error

    return beta_hat, gamma_inv_list, history


def parallel_align_batch_gpu(mu, betas, t, iterations=10, tol=10 ** (-5), n_jobs_temporal=-1, method="DP"):
    """Fast batched alignment: GPU for rotation/compose, CPU temporal reparam (fdasrsf).
    betas: list of N tensors (29, 3, 200). 
    method: 'DP' or 'RBFGS'. 
    Returns list(betas_aligned), list(gammas), list(histories)."""
    betas_stacked = torch.stack(betas, dim=0)
    beta_hat, gamma_inv_list, history = temporal_rotation_align_batch(
        mu, betas_stacked, t, iterations=iterations, tol=tol, n_jobs_temporal=n_jobs_temporal, method=method
    )
    betas_aligned = [beta_hat[n] for n in range(beta_hat.shape[0])]
    N = len(betas_aligned)
    return betas_aligned, gamma_inv_list, [history] * N


# def segment(beta, n=300):
#     """Split beta into segments of n timesteps along the last axis.
#     beta: array or tensor of shape (landmarks, 3, T).
#     Returns list of segments."""
#     T = beta.shape[2]
#     segments = []
#     start = 0
#     while start < T:
#         end = min(start + n, T)
#         seg = beta[..., start:end]
#         segments.append(seg)
#         start = end
#     return segments


def process_kinematic(data, gamma_t):
    pids = data.keys()
    betas_resampled = []

    for i, pid in enumerate(pids):
        beta = preprocess_temporal(data[pid])
        t = torch.linspace(0, 1, steps=beta.shape[2])

        beta_resampled = compose_gpu(torch.from_numpy(beta).to(device), t, gamma_t).cpu().numpy()
        betas_resampled.append(beta_resampled)

    return betas_resampled


def frechet(betas, t, mu_init, iterations=50, plot=True, tol=10 ** (-5), n_jobs_temporal=-1, method="DP"):
    """Frechet mean. method: 'DP' or 'RBFGS'. For speed try fewer iterations, looser tol, or method='RBFGS'."""
    betas_orig = np.copy(betas)

    epsilon = 0.1
    prev_error = -10000

    history = []

    N = len(betas)

    # Quotient translation and scaling
    # for n in range(N):
    # betas[n] = preprocess_temporal(betas[n])

    mu = mu_init
    mu = torch.from_numpy(mu).to(device)
    betas = [torch.from_numpy(beta).to(device) for beta in betas]
    t = torch.from_numpy(t).to(device)

    for iteration in tqdm(range(iterations)):
        betas_aligned, gammas, temp_histories = parallel_align_batch_gpu(mu, betas, t, n_jobs_temporal=n_jobs_temporal, method=method)

        betas_aligned_torch = torch.stack(betas_aligned, dim=0)

        # Compute all tangent vectors at once via batched log (same as log_gpu_frechet, one fewer permute)
        mu_batch = mu.unsqueeze(0).expand(N, -1, -1, -1)
        tangent_vec = log_gpu_batch(mu_batch, betas_aligned_torch)
        mean_tangent_vec = tangent_vec.mean(dim=0)

        # update mu
        mu = exp_gpu(mu, epsilon * mean_tangent_vec)

        error = torch.linalg.norm(mean_tangent_vec) ** 2
        error = error.item()
        history.append(error)

        if abs(error - prev_error) < tol:
            break
        prev_error = error
 
        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(history, marker='o', markersize=3)
            plt.xlabel('iteration')
            plt.ylabel('error')
            plt.title('Frechet mean: error curve')
            plt.show()

    mu = mu.cpu().numpy()
    betas_aligned = [beta.cpu().numpy() for beta in betas_aligned]
    # Return tangent_vec in same shape as before: (29, 3, 200, N)
    tangent_vec = tangent_vec.permute(1, 2, 3, 0).cpu().numpy()
    
    return mu, betas_aligned, gammas, tangent_vec, history

