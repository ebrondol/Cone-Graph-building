import argparse
import os.path as osp

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import awkward as ak
import networkx as nx

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from numba import jit
import os, errno
import uproot as uproot
import glob
import time
import sys
from tqdm import tqdm

def loadData(path, num_files = -1):
    """
    Loads pickle files of the graph data for network training.
    """
    f_edges_label = glob.glob(f"{path}*edges_labels.pkl")
    f_edges_features = glob.glob(f"{path}*edges_features.pkl")
    f_edges_scores_Ecp = glob.glob(f"{path}*edges_scores_1.pkl")
    f_edges_scores_Ereco = glob.glob(f"{path}*edges_scores_2.pkl")
    f_edges = glob.glob(f"{path}*edges.pkl" )
    f_nodes_features = glob.glob(f"{path}*node_features.pkl")
    f_best_simTs_match = glob.glob(f"{path}*_best_simTs_match.pkl")
    f_candidate_match = glob.glob(f"{path}*_candidate_match.pkl")
    
    edges_label, edges_scores_Ecp, edges_scores_Ereco, edges, nodes_features, best_simTs_match, candidate_match, edges_features = [], [], [], [], [], [], [], []
    n = len(f_edges_label) if num_files == -1 else num_files

    for i_f, _ in enumerate(tqdm(f_edges_label)):
        
        # Load the data
        if (i_f <= n):
            f = f_edges_label[i_f]
            with open(f, 'rb') as fb:
                edges_label.append(pickle.load(fb))
                
            f = f_edges_features[i_f]
            with open(f, 'rb') as fb:
                edges_features.append(pickle.load(fb))
                
            f = f_edges_scores_Ecp[i_f]
            with open(f, 'rb') as fb:
                edges_scores_Ecp.append(pickle.load(fb))
                
            f = f_edges_scores_Ereco[i_f]
            with open(f, 'rb') as fb:
                edges_scores_Ereco.append(pickle.load(fb))
                
            f = f_edges[i_f]
            with open(f, 'rb') as fb:
                edges.append(pickle.load(fb))
                
            f = f_nodes_features[i_f]
            with open(f, 'rb') as fb:
                nodes_features.append(pickle.load(fb))
                
            f = f_best_simTs_match[i_f]
            with open(f, 'rb') as fb:
                best_simTs_match.append(pickle.load(fb))
                
            f = f_candidate_match[i_f]
            with open(f, 'rb') as fb:
                candidate_match.append(pickle.load(fb))
                
        else:
            break
            
    return edges_label, edges_scores_Ecp, edges_scores_Ereco, edges, nodes_features, best_simTs_match, candidate_match, edges_features

@jit
def findNearestNeighbour(i, barycenters_x, barycenters_y, barycenters_z):
    # find nn, dist to nn for trackster i
    pos_i = np.array([barycenters_x[i], barycenters_y[i], barycenters_z[i]])
    d_least = 1000.
    for k in range(len(barycenters_x)):
        if k == i:
            continue
        pos_k = np.array([barycenters_x[k], barycenters_y[k], barycenters_z[k]])
        del_pos = pos_k - pos_i
        d = np.sqrt(del_pos[0]**2 + del_pos[1]**2 + del_pos[2]**2)
        if d < d_least:
            d_least = d
            i_least = k
    return i_least, d_least

def mkdir_p(mypath):
    '''Function to create a new directory, if it not already exist.
       Args:
           mypath : directory path
    '''
    from errno import EEXIST
    from os import makedirs, path
    try:
        makedirs(mypath)
    except OSError as exc:
        if not (exc.errno == EEXIST and path.isdir(mypath)):
            raise
            
def calculate_dist(lcs_i, lcs_j):
    # calculates distances between every pair of lcs in 2 tracksters
    dists = []
    for x_i, y_i, z_i in lcs_i:
        for x_j, y_j, z_j in lcs_j:
            dists.append(np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2))
    return dists


                
def set_small_to_zero(a, eps=1e-8):
    a[np.abs(a) < eps] = 0
    return a
                

            
def standardize_data(x_np):
    # WRONG at this point, look at the new features 
    
    mean, std = [], []
    
    # Barycenter coordinates x,y,z: 0,1,2
    # eVector0_x/y/z: 3,4,5
    # EV1/EV2/EV3: 6,7,8
    # sigmaPCA1/2/3: 9,10,11
    
    scaler = StandardScaler()
    
    x_full = x_np[:, :11]
    scaler.fit(x_full)
    x_norm = scaler.transform(x_full)
    mean.append(scaler.mean_)
    std.append(scaler.scale_)
    #print(f"Standardization constants \t mean: {scaler.mean_} \t var: {scaler.scale_}")
    
    # for the unnormalized features
    rest_num = x_np.shape[1] - 12
    assert rest_num == 3
    
    mean.append(np.zeros(rest_num)) 
    std.append(np.ones(rest_num))

    # Concatenate the normalized data
    x_norm = np.concatenate((x_norm, x_np[:, 12:]), axis=1)
    return x_norm, np.concatenate(mean, axis=-1), np.concatenate(std, axis=-1)
    
def prepare_test_data(data_list, ev):
    """
    Function to prepare (and possibly standardize) the test data
    """
    x_np, edge_label, edge_index, edge_score_Ecp, edge_score_Ereco, best_simTs_match, cand_match, edge_features = data_list[ev]
    #x_norm, mean, std = standardize_data(x_np)

    # Create torch vectors from the numpy arrays
    x = torch.from_numpy(x_np)
    x = torch.nan_to_num(x, nan=0.0)
    
    edge_label = torch.from_numpy(edge_label)
    edge_score_Ecp = torch.from_numpy(edge_score_Ecp)
    edge_score_Ereco = torch.from_numpy(edge_score_Ereco)
    best_simTs_match = torch.from_numpy(best_simTs_match)
    cand_match = torch.from_numpy(cand_match)
    edge_index = torch.from_numpy(edge_index)
    edge_features = torch.from_numpy(edge_features)
    
    data = Data(x=x, num_nodes=torch.tensor(x.shape[0]), edge_index=edge_index, edge_label=edge_label,
               edge_score_Ecp=edge_score_Ecp, edge_score_Ereco=edge_score_Ereco, best_simTs_match=best_simTs_match,
               candidate_match=cand_match, edge_features=edge_features)
    return data
            
def flatten_lists(el, es1, es2, ed, nd, bs, cm, ef):
    edge_label, edge_score_Ecp, edge_score_Ereco, edge_data, node_data, best_simTs_match, candidate_match, edge_features = [], [], [], [], [], [], [], []
    for i, X in enumerate(nd):
        for ev in range(len(X)):
                  
            if len(ed[i][ev]) == 0:
                print(f"Event {i}:{ev} has NO edges. Skipping.")
                continue # skip events with no edges
            elif X[ev].shape[1] <= 1:
                print(f"Event {i}:{ev} has {X[ev].shape[1]} nodes. Skipping.")
                continue
            else:
                edge_data.append(ed[i][ev])
                edge_label.append(el[i][ev])
                node_data.append(X[ev])
                edge_score_Ecp.append(es1[i][ev])
                edge_score_Ereco.append(es2[i][ev])
                best_simTs_match.append(bs[i][ev])
                candidate_match.append(cm[i][ev])
                edge_features.append(ef[i][ev])
                
    return edge_label, edge_score_Ecp, edge_score_Ereco, edge_data, node_data, best_simTs_match, candidate_match, edge_features
