def reco_to_sim_score_evaluation(dataset, predictions, thr, numevents=1000, PU=False):
    
    reco_sim_scores_list, best_sim_matches_list, pred_cluster_energies_list, truth_cluster_energies_list = [], [], [], []
    cand_sim_scores_list, cand_best_sim_matches_list, cand_energies_list = [], [], []
    
    for ev in tqdm(range(numevents)):
        event = dataset[ev]
        E = event.x[:, 16]
        cluster_labels_candidate = event.candidate_match
        truth_cluster_labels = event.best_simTs_match
        num_truth_clusters = int(max(truth_cluster_labels)+1)

        truth_clusters = [[] for i in range(num_truth_clusters)]
        for t, l in enumerate(truth_cluster_labels):
            truth_clusters[int(l)].append(t)
            
        clusters_candidate = [[] for i in range(int(max(cluster_labels_candidate))+1)]
        for ts, cand in enumerate(cluster_labels_candidate):
            clusters_candidate[int(cand)].append(ts)
            
        truth_cluster_energies = []
        for t_cluster in truth_clusters:
            clusterE = 0
            for t in t_cluster:
                clusterE += E[t].item()
            truth_cluster_energies.append(clusterE)
        
        predicted_clusters, _ = find_connected_components(event, predictions[ev], edge_index=None, thr=thr, PU=PU)
        
        reco_sim_scores, best_sim_matches, pred_cluster_energies = scores_reco_to_sim(predicted_clusters, truth_cluster_labels, truth_cluster_energies, E.tolist())
        cand_sim_scores, cand_best_sim_matches, cand_energies = scores_reco_to_sim(clusters_candidate,  truth_cluster_labels, truth_cluster_energies, E.tolist())
        
#         print(f"RECO-SIM scores : {reco_sim_scores}")
#         print(f"Best SIM matches : {best_sim_matches}")

#         print(f"Predicted cluster energies (GeV) : {pred_cluster_energies}")
#         print(f"Truth cluster energies (GeV) : {truth_cluster_energies}\n")

        reco_sim_scores_list.append(reco_sim_scores)
        best_sim_matches_list.append(best_sim_matches)
        pred_cluster_energies_list.append(pred_cluster_energies)
        truth_cluster_energies_list.append(truth_cluster_energies)
        cand_sim_scores_list.append(cand_sim_scores)
        cand_best_sim_matches_list.append(cand_best_sim_matches)
        cand_energies_list.append(cand_energies)
        
    return reco_sim_scores_list, best_sim_matches_list, pred_cluster_energies_list, truth_cluster_energies_list, cand_sim_scores_list, cand_best_sim_matches_list, cand_energies_list

# Validation score between every "super trackster" and simtrackster a trackster is "in" it's best matched simtrackster
# Calculated as intersection / union; 1 = perfect match 
# Intersection = sum(Energy of tracksters common)
# for every "super trackster" consider only the best score

    
#@jit
def scores_reco_to_sim(predicted_clusters, truth_cluster_labels, truth_cluster_energies, E):
    num_truth_clusters = int(max(truth_cluster_labels)+1)
    pred_cluster_energies = []
    reco_sim_scores = []
    best_sim_matches = []
    
    for pred_cluster in predicted_clusters:
        # for each predicted cluster, one score for every truth cluster
        scores_cluster = np.zeros(num_truth_clusters) # [0 for i in range(num_truth_clusters)]
        clusterE = 0
        for trackster in pred_cluster:
            truth_label = int(truth_cluster_labels[trackster])
            scores_cluster[truth_label] += E[trackster]
            clusterE += E[trackster]
        pred_cluster_energies.append(clusterE)
        
        for cluster in range(num_truth_clusters):
            scores_cluster[cluster] /= (pred_cluster_energies[-1] + truth_cluster_energies[cluster] - scores_cluster[cluster])
        reco_sim_scores.append(np.max(scores_cluster))
        best_sim_matches.append(np.argmax(scores_cluster))
        
    return reco_sim_scores, best_sim_matches, pred_cluster_energies


def getAccuracy(y_true, y_prob, classification_threshold):
    # TODO: make this a vector function
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > classification_threshold
    return (y_true == y_prob).sum().item() / y_true.size(0)

def connectivity_matrix(model, data_list, ev, thr, similarity=True, prepare_network_input_data=None):
    data_ev = prepare_test_data(data_list, ev)
    
    if prepare_network_input_data is not None:
        inputs = prepare_network_input_data(data_ev.x, data_ev.edge_index, 
                                            device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    else:
        inputs = data_ev.x, data_ev.edge_index
    out, emb = model(*inputs)
    N = data_ev.num_nodes
    mat = np.zeros([N, N])
    truth_mat = np.zeros([N, N])
    for indx, src in enumerate(data_ev.edge_index[0]):
        dest = data_ev.edge_index[1][indx]
        mat[src][dest] = out[indx]
        mat[dest][src] = out[indx]
        truth_mat[src][dest] = data_ev.edge_label[indx]
        truth_mat[dest][src] = data_ev.edge_label[indx]
        
    if similarity == False:
        mat = mat > thr
    return mat, truth_mat