import numpy as np
import torch
import ripserplusplus as rpp

def sample_W(W, nSamples, isRandom=True):
    n = W.shape[0]
    random_indices = np.random.choice(n, size=nSamples, replace=False)
    return W[random_indices]

def sum_intervals_ripser(embeddings, h_dim=0):    
    diagrams = rpp.run("--format point-cloud", embeddings)
    #persistence["ripser_sum"] = 0
    # Compute condensed pairwise distances (1D array)
    # Convert to square distance matrix
    persistence_sum = 0.0
    
    if len(diagrams) > h_dim:
        persistence_sum = sum([death - birth for birth, death in diagrams[h_dim] if death > birth])
    return persistence_sum


def calculate_ph_dim(W, min_points_in=200, max_points_in=1000, point_jump_in=50,  
        h_dim=0, print_error=False, **kwargs):
    # sample_fn should output a [num_points, dim] array
    if W.shape[0] < max_points_in:
        min_points = max(10, int((float(W.shape[0]) / max_points_in) * min_points_in))
        max_points = W.shape[0]
        point_jump = max(1, int((float(W.shape[0]) / max_points_in) * point_jump_in))
    else:
        min_points, max_points, point_jump = min_points_in, max_points_in, point_jump_in
        
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    
    for n in test_n:
        curr_l = sum_intervals_ripser(sample_W(W, n), h_dim=h_dim)  
        lengths.append(curr_l)
        
    lengths = np.array(lengths)
    
    # compute our ph dim by running a linear least squares
    x = np.log(np.array(list(test_n)))
    y = np.log(lengths)
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)

def calculate_ph_dim_gpu(W, min_points=200, max_points=1000, 
        point_jump=50, h_dim=0, print_error=False):
    from torchph.pershom import vr_persistence
    # sample_fn should output a [num_points, dim] array
    
    # sample our points
    test_n = range(min_points, max_points, point_jump)
    lengths = []
    for n in test_n:
        samples = sample_W(W, n)
        dist_matrix = torch.cdist(samples, samples)
        
        d, _ = vr_persistence(dist_matrix, 0, 0)
        d = d[0]
        lengths.append((d[:, 1] - d[:, 0]).sum())

    lengths = torch.stack(lengths)
    
    # compute our ph dim by running a linear least squares
    x = torch.tensor(test_n).to(lengths).log()
    y = lengths.log()
    N = len(x)
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = y.mean() - m * x.mean()
    
    error = ((y - (m * x + b)) ** 2).mean()
    
    if print_error:
        print(f"Ph Dimension Calculation has an approximate error of: {error}.")
    return 1 / (1 - m)