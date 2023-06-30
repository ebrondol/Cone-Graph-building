import argparse
import os.path as osp
import pdb
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import awkward as ak

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
    #f_edges_features = glob.glob(f"{path}*edges_features.pkl")
    f_edges_scores_Ecp = glob.glob(f"{path}*edges_scores_1.pkl")
    f_edges_scores_Ereco = glob.glob(f"{path}*edges_scores_2.pkl")
    f_edges = glob.glob(f"{path}*edges.pkl" )
    f_nodes_features = glob.glob(f"{path}*node_features.pkl")
    f_best_simTs_match = glob.glob(f"{path}*_best_simTs_match.pkl")
    f_candidate_match = glob.glob(f"{path}*_candidate_match.pkl")

    f_SC_energy = glob.glob(f"{path}*_SC_energy.pkl")
    f_SC_pid = glob.glob(f"{path}*_SC_pid.pkl")
    f_SC_ass_tracksters = glob.glob(f"{path}*_SC_ass_tracksters.pkl")

    
    edges_label, edges_scores_Ecp, edges_scores_Ereco, edges, nodes_features, best_simTs_match, candidate_match, edges_features = [], [], [], [], [], [], [], []
    SC_energy, SC_pid, SC_ass_tracksters = [],[],[]
    
    n = len(f_edges_label) if num_files == -1 else num_files

    for i_f, _ in enumerate(tqdm(f_edges_label)):
        
        # Load the data
        if (i_f <= n):
            f = f_edges_label[i_f]
            with open(f, 'rb') as fb:
                edges_label.append(pickle.load(fb))
                
            #f = f_edges_features[i_f]
            #with open(f, 'rb') as fb:
            #    edges_features.append(pickle.load(fb))
                
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

            f = f_SC_energy[i_f]
            with open(f, 'rb') as fb:
                SC_energy.append(pickle.load(fb))

            f = f_SC_pid[i_f]
            with open(f, 'rb') as fb:
                SC_pid.append(pickle.load(fb))

            f = f_SC_ass_tracksters[i_f]
            with open(f, 'rb') as fb:
                SC_ass_tracksters.append(pickle.load(fb))
                
        else:
            break
            
    return edges_label, edges_scores_Ecp, edges_scores_Ereco, edges, nodes_features, best_simTs_match, candidate_match, SC_energy,SC_pid,SC_ass_tracksters#, edges_features

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


def computeEdgeAndLabels(trk_data, ass_data, gra_data, edges, edges_labels, edge_scores_1, edge_scores_2, 
                         simtrackstersSC_data, best_simTs_match, score_threshold = 0.2):
    '''Function to compute the truth graph and assign the edge scores and labels.
    '''            
    for i in range(trk_data.NTracksters):
        # Calculating the quality reco to sim scores
        qualities_i = ass_data.tsCLUE3D_recoToSim_SC_score[i]
        # Getting the best (minimum) score value
        score_i = ak.min(qualities_i)
        # Getting the index of the best (minimum) score value = the index of the best-assigned simTrackster
        best_sts_i = ak.argmin(qualities_i)
        # Converting the index of the best-assigned simTrackster to the SimTrackster id
        best_sts_i_index = ass_data.tsCLUE3D_recoToSim_SC[i][best_sts_i]
        best_sts_i_index = best_sts_i_index if score_i < score_threshold else -1

        # If the quality score is higher than the threshold, we say that there is no 
        # best-assigned simTrackster and set `best_sts_i` to -1 (non-existent)
        best_sts_i = best_sts_i if score_i < score_threshold else -1

        # Appending the list of the `best_simTs_match` which is used as one of the training properties
        best_simTs_match.append(best_sts_i_index)

        # If there is an assigned SimTrackstor:
        if best_sts_i != -1:
            # Getting the recoTrackster raw energy
            tr_i_en = trk_data.raw_energy[i]
            # Getting the shared energy between the recoTrackster and the simTrackster
            shared_en_i = ass_data.tsCLUE3D_recoToSim_SC_sharedE[i][best_sts_i]
            # Calculating a non-normalized energy score
            energy_score_i = shared_en_i*(1-score_i)
            # Energy of the simTrackster
            SC_E = simtrackstersSC_data.stsSC_raw_energy[best_sts_i_index]

            # For every linked (inner) trackster:
            for j in gra_data.linked_inners[i]:
                # Create an edge between the tracksters
                edges.append([j,i])
                # Calculating the quality reco to sim scores
                qualities_j = ass_data.tsCLUE3D_recoToSim_SC_score[j]
                # Getting the best (minimum) score value
                score_j = ak.min(qualities_j)     
                # Getting the index of the best (minimum) score value = the index of the best-assigned simTrackster
                best_sts_j = ak.argmin(qualities_j)
                best_sts_j_index = ass_data.tsCLUE3D_recoToSim_SC[j][best_sts_j]

                # If the quality score is higher than the threshold, we say that there is no 
                # best-assigned simTrackster and set `best_sts_i` to -1 (non-existent)
                best_sts_j = best_sts_j if score_j < score_threshold else -1
                best_sts_j_index = best_sts_j_index if score_j < score_threshold else -1

                if best_sts_i_index == best_sts_j_index and best_sts_i != -1:
                    # There is an edge -> label = 1
                    edges_labels.append(1)               
                    # Getting the recoTrackster raw energy
                    tr_j_en = trk_data.raw_energy[j]
                    # Getting the shared energy between the recoTrackster and the simTrackster
                    shared_en_j = ass_data.tsCLUE3D_recoToSim_SC_sharedE[j][best_sts_j]

                    energy_score_j = shared_en_j*(1-score_j)
                    edge_score_1 = (energy_score_i + energy_score_j)/SC_E
                    edge_score_2 = (energy_score_i/tr_i_en + energy_score_j/tr_j_en)/2
                    edge_scores_1.append(edge_score_1)  
                    edge_scores_2.append(edge_score_2)  

        #                 assert energy_score_i != 0 and energy_score_j != 0 and edge_score_1 <= 1, f"Problem with scores!\n energy_score_i: {energy_score_i}, energy_score_j: {energy_score_j}, edge_score_1: {edge_score_1}, edge_score_2: {edge_score_2}; \n Trackster {i} assigned to SimTrackster {best_sts_i} with score {score_i};\n Trackster energy is {tr_i_en}, shared energy {shared_en_i}, SC energy {SC_E} and edge score {energy_score_i / SC_E}\n Trackster {j} assigned to SimTrackster {best_sts_j} with score {score_j};\n Trackster energy is {tr_j_en}, shared energy {shared_en_j}, SC energy {SC_E} and edge score {energy_score_j / SC_E}\n\n"


                else:
                    # There is no edge -> label = 0 and both scores are also 0
                    edges_labels.append(0)
                    edge_scores_1.append(0)
                    edge_scores_2.append(0)

                    # Stop connecting trackster with best_sts_i = -1 (no connection to the simTrackster),
                    # as it already has one connection in graph
                    if best_sts_i == -1:
                        break


            if (len(gra_data.linked_inners[i]) == 0 and len(gra_data.linked_outers[i]) == 0):
        #             if len(gra_data.linked_inners[i]) == 0 and len(gra_data.linked_outers[i]) == 0:
        #                 print(f"Trackster does not have any neighbours in the graph, connecting it to its nearest neighbour.")

                # this trackster does not have any neighbours in the graph, connect it to its nearest neighbour
                b_x = ak.to_numpy(trk_data.barycenter_x)
                b_y = ak.to_numpy(trk_data.barycenter_y)
                b_z = ak.to_numpy(trk_data.barycenter_z)
                nearest_id, nearest_dist = findNearestNeighbour(i, b_x, b_y, b_z)
                edges.append([i, nearest_id])
                edges_labels.append(0)
                edge_scores_1.append(0)
                edge_scores_2.append(0)
                
# @jit
# def computeEdgeAndLabels(trk_data, ass_data, gra_data, edges, edges_labels, edge_scores_1, edge_scores_2, 
#                          simtrackstersSC_data, best_simTs_match, score_threshold = 0.2):
#     '''Function to compute the truth graph and assign the edge scores and labels.
#     '''
#     for i in range(trk_data.NTracksters):
#         # Calculating the quality reco to sim scores
#         qualities_i = ass_data.tsCLUE3D_recoToSim_SC_score[i]
#         # Getting the best (minimum) score value
#         score_i = ak.min(qualities_i)
#         # Getting the index of the best (minimum) score value = the index of the best-assigned simTrackster
#         best_sts_i = ak.argmin(qualities_i)
#         # Converting the index of the best-assigned simTrackster to the SimTrackster id
#         best_sts_i_index = ass_data.tsCLUE3D_recoToSim_SC[i][best_sts_i]
        
#         # If the quality score is higher than the threshold, we say that there is no 
#         # best-assigned simTrackster and set `best_sts_i` to -1 (non-existent)
#         #best_sts_i = best_sts_i if score_i < score_threshold else -1
        
#         # Appending the list of the `best_simTs_match` which is used as one of the training properties
#         best_simTs_match.append(best_sts_i_index)
            
#         # If there is an assigned SimTrackstor:
#         if best_sts_i != -1:
#             # Getting the recoTrackster raw energy
#             tr_i_en = trk_data.raw_energy[i]
#             # Getting the shared energy between the recoTrackster and the simTrackster
#             shared_en_i = ass_data.tsCLUE3D_recoToSim_SC_sharedE[i][best_sts_i]
#             # Calculating a non-normalized energy score
#             energy_score_i = shared_en_i*(1-score_i)
#             # Energy of the simTrackster
#             SC_E = simtrackstersSC_data.stsSC_raw_energy[best_sts_i_index]
        
#         # For every linked (inner) trackster:
#         for j in gra_data.linked_inners[i]:
#             # Create an edge between the tracksters
#             edges.append([j,i])
#             # Calculating the quality reco to sim scores
#             qualities_j = ass_data.tsCLUE3D_recoToSim_SC_score[j]
#             # Getting the best (minimum) score value
#             score_j = ak.min(qualities_j)     
#             # Getting the index of the best (minimum) score value = the index of the best-assigned simTrackster
#             best_sts_j = ak.argmin(qualities_j)
#             best_sts_j_index = ass_data.tsCLUE3D_recoToSim_SC[j][best_sts_j]
            
#             # If the quality score is higher than the threshold, we say that there is no 
#             # best-assigned simTrackster and set `best_sts_i` to -1 (non-existent)
#             best_sts_j = best_sts_j if score_j < score_threshold else -1

#             if best_sts_i_index == best_sts_j_index and best_sts_i != -1:
#                 # There is an edge -> label = 1
#                 edges_labels.append(1)               
#                 # Getting the recoTrackster raw energy
#                 tr_j_en = trk_data.raw_energy[j]
#                 # Getting the shared energy between the recoTrackster and the simTrackster
#                 shared_en_j = ass_data.tsCLUE3D_recoToSim_SC_sharedE[j][best_sts_j]
                
#                 energy_score_j = shared_en_j*(1-score_j)
#                 edge_score_1 = (energy_score_i + energy_score_j)/SC_E
#                 edge_score_2 = (energy_score_i/tr_i_en + energy_score_j/tr_j_en)/2
#                 edge_scores_1.append(edge_score_1)  
#                 edge_scores_2.append(edge_score_2)  
                
# #                 assert energy_score_i != 0 and energy_score_j != 0 and edge_score_1 <= 1, f"Problem with scores!\n energy_score_i: {energy_score_i}, energy_score_j: {energy_score_j}, edge_score_1: {edge_score_1}, edge_score_2: {edge_score_2}; \n Trackster {i} assigned to SimTrackster {best_sts_i} with score {score_i};\n Trackster energy is {tr_i_en}, shared energy {shared_en_i}, SC energy {SC_E} and edge score {energy_score_i / SC_E}\n Trackster {j} assigned to SimTrackster {best_sts_j} with score {score_j};\n Trackster energy is {tr_j_en}, shared energy {shared_en_j}, SC energy {SC_E} and edge score {energy_score_j / SC_E}\n\n"

                    
#             else:
#                 # There is no edge -> label = 0 and both scores are also 0
#                 edges_labels.append(0)
#                 edge_scores_1.append(0)
#                 edge_scores_2.append(0)
                
#                 # Stop connecting trackster with best_sts_i = -1 (no connection to the simTrackster),
#                 # as it already has one connection in graph
#                 if best_sts_i == -1:
#                     break

        
#         if (len(gra_data.linked_inners[i]) == 0 and len(gra_data.linked_outers[i]) == 0):
# #             if len(gra_data.linked_inners[i]) == 0 and len(gra_data.linked_outers[i]) == 0:
# #                 print(f"Trackster does not have any neighbours in the graph, connecting it to its nearest neighbour.")
                
#             # this trackster does not have any neighbours in the graph, connect it to its nearest neighbour
#             b_x = ak.to_numpy(trk_data.barycenter_x)
#             b_y = ak.to_numpy(trk_data.barycenter_y)
#             b_z = ak.to_numpy(trk_data.barycenter_z)
#             nearest_id, nearest_dist = findNearestNeighbour(i, b_x, b_y, b_z)
#             edges.append([i, nearest_id])
#             edges_labels.append(0)
#             edge_scores_1.append(0)
#             edge_scores_2.append(0)
                
                
def save_pickle_dataset(input_folder, outputPath, offset = 0, jupyter=False):
    """
    Saves the simulation data in pickle files.
    """
    if jupyter: from tqdm.autonotebook import tqdm
    else: from tqdm import tqdm
        
    print(f"Folder : {input_folder}")
    files = glob.glob(f"{input_folder}/*ntuples_*.root")
    print(f"Number of files: {len(files)}")

    X, Edges, Edges_labels, Edges_scores_1, Edges_scores_2, Best_simTs_match, Candidate_match = [], [], [], [], [], [], []
    SC_energy, SC_pid, SC_ass_tracksters = [], [], []

    mkdir_p(outputPath)
    cum_events = 0

    N = 10e6
    
    for i_file, file in enumerate(files[offset:]):
        i_file += offset
        if i_file >= N: break
        try:
            with uproot.open(file) as f:
                t =  f["ticlNtuplizer/tracksters"]
                calo = f["ticlNtuplizer/simtrackstersSC"]
                ass = f["ticlNtuplizer/associations"]
                gra = f["ticlNtuplizer/graph"]
                simtrackstersSC = f["ticlNtuplizer/simtrackstersSC"]
                cand = f["ticlNtuplizer/candidates"]

                trk_data = t.arrays(["NTracksters", "raw_energy","raw_em_energy","barycenter_x","barycenter_y",
                                     "barycenter_z","eVector0_x", "eVector0_y","eVector0_z","EV1","EV2","EV3",
                                     "vertices_indexes", "sigmaPCA1", "sigmaPCA2", "sigmaPCA3"])
                gra_data = gra.arrays(["linked_inners", "linked_outers"])
                ass_data = ass.arrays(["tsCLUE3D_recoToSim_SC", "tsCLUE3D_recoToSim_SC_score",
                                       "tsCLUE3D_simToReco_SC", "tsCLUE3D_simToReco_SC_score",
                                       "tsCLUE3D_simToReco_SC_sharedE", "tsCLUE3D_recoToSim_SC_sharedE"])
                simtrackstersSC_data = simtrackstersSC.arrays(["stsSC_raw_energy","stsSC_regressed_energy","stsSC_pdgID"])
                #simtrackstersSC_data = simtrackstersSC.arrays("stsSC_raw_energy")
                cand_data = cand.arrays(["tracksters_in_candidate"])

        except Exception as e:
            print(f"Error: {e}")
            continue

        print('\nProcessing file {} '.format(file), end="")
        cum_events += len(gra_data)
        if(cum_events%1000 == 0):
            print(f"\nNumber of events processed: {cum_events}")

        start = time.time()
        for ev in tqdm(range(len(gra_data))):
           

            trackster_sizes = []
            for vertices in trk_data[ev].vertices_indexes:
                trackster_sizes.append(ak.size(vertices))
                
            in_candidate = [-1 for i in range(trk_data[ev].NTracksters)]
            for indx, cand in enumerate(cand_data[ev].tracksters_in_candidate):
                for ts in cand:
                    in_candidate[ts] = indx


            edges, edges_labels, edge_scores_1, edge_scores_2, best_simTs_match = [], [], [], [], []
            
            if len(trk_data[ev].barycenter_x) <= 1:
                # Skip events with less than one trackster
                continue
            
            computeEdgeAndLabels(trk_data[ev], ass_data[ev], gra_data[ev], edges, edges_labels, edge_scores_1,
                                 edge_scores_2, simtrackstersSC_data[ev], best_simTs_match)
            
            # Save the input variables
            x_ev = np.array([
                   trk_data[ev].barycenter_x,
                   trk_data[ev].barycenter_y,
                   trk_data[ev].barycenter_z,
                   trk_data[ev].eVector0_x,
                   trk_data[ev].eVector0_y,
                   trk_data[ev].eVector0_z,
                   trk_data[ev].EV1,
                   trk_data[ev].EV2,
                   trk_data[ev].EV3,
                   trk_data[ev].sigmaPCA1,
                   trk_data[ev].sigmaPCA2,
                   trk_data[ev].sigmaPCA3,
                   trackster_sizes,
                   trk_data[ev].raw_energy, 
                   trk_data[ev].raw_em_energy], dtype=np.float32)

            if len(edges) == 0:
                print(f"Event {ev} has zero edges. Skipping.")
                continue # Skip events with no edges

            X.append(x_ev)
            Edges.append(np.array(edges).T)
            Edges_labels.append(np.array(edges_labels))
            Edges_scores_1.append(np.array(edge_scores_1, dtype=np.float32))
            Edges_scores_2.append(np.array(edge_scores_2, dtype=np.float32))
            Best_simTs_match.append(np.array(best_simTs_match, dtype=np.float32))
            Candidate_match.append(np.array(in_candidate, dtype=np.float32))
            SC_pid.append(np.array(simtrackstersSC_data[ev].stsSC_pdgID))
            SC_energy.append(np.array(simtrackstersSC_data[ev].stsSC_regressed_energy))
            SC_ass_tracksters.append(np.array(ass_data[ev].tsCLUE3D_simToReco_SC))

            ev_id = ev + 1
            # Save to disk
            if((ev_id % 500 == 0 and ev_id != 1)  or (ev_id == len(gra_data))):
                stop = time.time()
                print(f"t = {stop-start} ... Saving the pickle data for {i_file}_{ev_id}")

                with open(f"{outputPath}{i_file}_{ev_id}_node_features.pkl", "wb") as fp:
                    # Node properties
                    pickle.dump(X, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_edges.pkl", "wb") as fp: 
                    # Graph edges
                    pickle.dump(Edges, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_edges_labels.pkl", "wb") as fp:
                    # Edge labels (is this a true edge -> 1; the edge should not be there -> 0)
                    pickle.dump(Edges_labels, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_edges_scores_1.pkl", "wb") as fp:
                    pickle.dump(Edges_scores_1, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_edges_scores_2.pkl", "wb") as fp:
                    pickle.dump(Edges_scores_2, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_best_simTs_match.pkl", "wb") as fp:
                    pickle.dump(Best_simTs_match, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_candidate_match.pkl", "wb") as fp:
                    pickle.dump(Candidate_match, fp)
                    
                # SimTrackster Truth Information
                with open(f"{outputPath}{i_file}_{ev_id}_SC_energy.pkl", "wb") as fp:
                    pickle.dump(SC_energy, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_SC_pid.pkl", "wb") as fp:
                    pickle.dump(SC_pid, fp)
                with open(f"{outputPath}{i_file}_{ev_id}_SC_ass_tracksters.pkl", "wb") as fp:
                    pickle.dump(SC_ass_tracksters, fp)
                    
                # Scores statistics printing
                Edges_scores_flat = [item for sublist in Edges_scores_1 for item in sublist]
                print(f"Mean edge score 1: {np.mean(Edges_scores_flat)}  Min: {min(Edges_scores_flat)}  Max: {max(Edges_scores_flat)}")
                
                Edges_scores_flat = [item for sublist in Edges_scores_2 for item in sublist]
                print(f"Mean edge score 2: {np.mean(Edges_scores_flat)}  Min: {min(Edges_scores_flat)}  Max: {max(Edges_scores_flat)}")
                
                # Emptying arrays
                ed_np, X, Edges, Edges_labels, Edges_scores_1, Edges_scores_2, Best_simTs_match, Candidate_match = [], [], [], [], [], [], [], []
                start = time.time()
            
def standardize_data(x_np):
    
    mean = []
    std = []
    
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
    Function to prepare (standardize) the test data
    """
    x_np, edge_label, edge_index, edge_score_Ecp, edge_score_Ereco, best_simTs_match, cand_match = data_list[ev]
    #x_norm, mean, std = standardize_data(x_np)

    # Create torch vectors from the numpy arrays
    x = torch.from_numpy(x_np)
    edge_label = torch.from_numpy(edge_label)
    edge_score_Ecp = torch.from_numpy(edge_score_Ecp)
    edge_score_Ereco = torch.from_numpy(edge_score_Ereco)
    best_simTs_match = torch.from_numpy(best_simTs_match)
    cand_match = torch.from_numpy(cand_match)
    edge_index = torch.from_numpy(edge_index)
    
    data = Data(x=x, num_nodes=torch.tensor(x.shape[0]), edge_index=edge_index, edge_label=edge_label,
               edge_score_Ecp=edge_score_Ecp, edge_score_Ereco=edge_score_Ereco, best_simTs_match=best_simTs_match,
               candidate_match=cand_match)
    return data
            
def flatten_lists(el, es1, es2, ed, nd, bs, cm, se, sp, sat):
    edge_label, edge_score_Ecp, edge_score_Ereco, edge_data, node_data, best_simTs_match, candidate_match, sc_energy, sc_pid, sc_ass_tracksters = [], [], [], [], [], [], [], [], [], []
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
                sc_energy.append(se[i][ev])
                sc_pid.append(sp[i][ev])
                sc_ass_tracksters.append(sat[i][ev])
    #flatten_list = [ele for inner_list in pkl_list for ele in inner_list]
    return edge_label, edge_score_Ecp, edge_score_Ereco, edge_data, node_data, best_simTs_match, candidate_match, sc_energy, sc_pid, sc_ass_tracksters
             
    
def save_dataset(pickle_data, output_location, trainRatio = 0.8, valRatio = 0.1, testRatio = 0.1):
        
    print("Loading Pickle Files...")
    # obtain edges_label, edges, nodes_features from all the pickle files
    el, es1, es2, ed, nd, bs, cm, se, sp, sat = loadData(pickle_data, num_files = -1)
    print("Loaded.")
    
#     edge_label = flatten_list(edge_label)
#     edge_score_Ecp = flatten_list(edge_score_Ecp)
#     edge_score_Ereco = flatten_list(edge_score_Ereco)
#     edge_data = flatten_list(edge_data)
#     node_data = flatten_list(node_data)
#     best_simTs_match = flatten_list(best_simTs_match)
#     candidate_match = flatten_list(candidate_match)
    edge_label, edge_score_Ecp, edge_score_Ereco, edge_data, node_data, best_simTs_match, candidate_match, sc_energy, sc_pid, sc_ass_tracksters = flatten_lists(el, es1, es2, ed, nd, bs, cm, se, sp, sat)

    data_list = []
    print(f"{len(node_data)} total events in dataset.")

    nSamples = len(node_data)
    nTrain = int(trainRatio * nSamples)
    nVal = int(valRatio * nSamples)

    print("Preparing training and validation split")
    for ev in tqdm(range(len(node_data[:nTrain+nVal]))):
                
        x_np = node_data[ev].T
        #x_norm, _, _ = standardize_data(x_np)
        
        # Create torch vectors from the numpy arrays
        x = torch.from_numpy(x_np)
        e_label = torch.from_numpy(edge_label[ev])
        e_score_Ecp = torch.from_numpy(edge_score_Ecp[ev])
        e_score_Ereco = torch.from_numpy(edge_score_Ereco[ev])
        edge_index = torch.from_numpy(edge_data[ev])
        b_simTs_match = torch.from_numpy(best_simTs_match[ev])
        cand_match = torch.from_numpy(candidate_match[ev])
        sc_e = torch.from_numpy(sc_energy[ev])
        sc_p = torch.from_numpy(sc_pid[ev])
        sc_at = torch.from_numpy(sc_ass_tracksters[ev])

        data = Data(x=x, num_nodes=torch.tensor(x.shape[0]),
                    edge_index=edge_index, edge_label=e_label, 
                    edge_score_Ecp=e_score_Ecp, edge_score_Ereco=e_score_Ereco,
                    best_simTs_match=b_simTs_match, candidate_match=cand_match, 
                    sc_energy=sc_e,sc_pid=sc_p, sc_ass_tracksters=sc_at)
        
        # This graph is directed.
        #print(f"data is directed: {data.is_directed()}")
        data_list.append(data)

    # The test split is not normalized and is stored as a list
    test_data_list = []
    
    print("Preparing test split (data not preprocessed)")
    for ev in tqdm(range(len(node_data[nTrain+nVal:]))):

        x_np = node_data[ev].T
        # Do not pre-process the test split
        data = [x_np, edge_label[ev], edge_data[ev], edge_score_Ecp[ev], edge_score_Ereco[ev],
               best_simTs_match[ev], candidate_match[ev]]
        test_data_list.append(data)


    trainDataset = data_list[:nTrain] # training dataset
    valDataset = data_list[nTrain:]   # validation dataset
    
    # Saves the dataset objects to disk.
    mkdir_p(f'{output_location}')
    torch.save(trainDataset, f'{output_location}/dataTraining.pt')
    torch.save(valDataset, f'{output_location}/dataVal.pt')
    torch.save(test_data_list, f'{output_location}/dataTest.pt')
    print("Done: Saved the training datasets.")
        
        
        
if __name__ == "__main__":
    
    datasets = ['pion', 'multiparticle', 'pionPU']
    parser = argparse.ArgumentParser(description='Preprocess graph data for HGCAL.')
    parser.add_argument('--dataset_name', type=str, default="pion", choices=datasets,
                    help='Dataset name to be processed')
    parser.add_argument('--offset', type=int, default=0, help='Offset of the file to start processing from (useful when the process fails).')
    parser.add_argument('--save_pkl', default=False, action='store_true')
    parser.add_argument('--save_dataset', default=False, action='store_true')
    args = parser.parse_args()
    
    dataset = args.dataset_name
    offset = args.offset
    dataset_id = datasets.index(dataset)

    if dataset_id==0:
        # Pion
        pkl_path = "/eos/user/e/ebrondol/SWAN_projects/Cone-Graph-building2/dataproduction/closeByDoublePion_pkl/test_3/"
        processed_dataset_path = "/eos/user/e/ebrondol/SWAN_projects/Cone-Graph-building2/dataproduction/closeByDoublePion_dataset/test_3/"
        n_tuples_path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv4Sample/Uniform10_600/CloseByTwoPion_fix2/ntuples_10_600/ntuples_10_600"

    if dataset_id==1:
        # Multiparticle
        pkl_path = "/eos/home-m/mmatthew/Patatrack13/Cone-Graph-building/dataproduction/Multiparticle_pkl_dataset_final_cand_fixed_no_norm/"
        processed_dataset_path = "/eos/home-m/mmatthew/Patatrack13/Cone-Graph-building/dataproduction/Multiparticle_dataset_final_cand_fixed_no_norm/"
        n_tuples_path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv4Sample/CloseBySamples/MultiParticle/ntuples_10_600"

    if dataset_id==2:
        pkl_path = "/eos/home-m/mmatthew/Patatrack13/Cone-Graph-building/dataproduction/PionPU_pkl_dataset_no_norm/"
        processed_dataset_path = "/eos/home-m/mmatthew/Patatrack13/Cone-Graph-building/dataproduction/PionPU_dataset_no_norm/"
        n_tuples_path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/wredjeb/TICLv4Sample/GNNTraining/SinglePion200PU/ntuples_10_600"

    if args.save_pkl:
        save_pickle_dataset(input_folder=n_tuples_path, outputPath=pkl_path, offset=offset)
    if args.save_dataset:
        save_dataset(pickle_data=pkl_path, output_location=processed_dataset_path)
    print("Finished")
    
