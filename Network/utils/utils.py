import matplotlib.pyplot as plt

from collections import defaultdict
import torch
import pdb


def get_regressed_sc_energy(pred_trk_energy,sc_trk_match,sc_energy):
    pred_sc_energy = torch.tensor([])
    for i in range(len(sc_energy)):
        idx = torch.where(sc_trk_match==i,1,0).nonzero()
        if pred_sc_energy.numel()==0:
            pred_sc_energy = pred_trk_energy[idx].sum().unsqueeze(dim=0)
        else:
            pred_sc_energy = torch.cat((pred_sc_energy,pred_trk_energy[idx].mean().unsqueeze(dim=0)),dim=0)
    return pred_sc_energy


def list_flatten(my_list):
    return [item for sublist in my_list for item in sublist]

def find_connected_components(event, predictions, edge_index=None, thr=0.9, PU=False):
    """
    if PU is false, it doesn't assign each -1 to a separate cluster
    """
    components = []
    predicted_cluster_ids = np.array([-1 for i in range(event.x.shape[0])])
    
    graph = defaultdict(set)
    i = 0
    # only for nodes that are connected with an edge, normally would also have to add nodes that are not connected to anything
    if edge_index == None:
        edge_index = event.edge_index.cpu().numpy().T
    else:
        edge_index = edge_index.cpu().numpy().T
    for u, v in edge_index:
        if predictions[i] >= thr:
            graph[u].add(v)
            graph[v].add(u)
        i += 1
    
    j = 0
    for component in connected_components(graph):
        c = list(set(component))
        components.append(c)
        predicted_cluster_ids[c] = j
        j += 1
        
    #print(components)
    #print(predicted_cluster_ids)
    if not PU:
        unassigned = np.where(predicted_cluster_ids == -1)[0]
        #print(unassigned)
        max_cl_id = np.max(predicted_cluster_ids)
        predicted_cluster_ids[unassigned] = range(max_cl_id+1, max_cl_id+unassigned.shape[0]+1) 
        for un_id in unassigned:
            components.append([un_id])
    #print(components)
    #print(predicted_cluster_ids)
    #print("----")
    return components, predicted_cluster_ids

def connected_components(neighbors):
    seen = set()
    def component(node):
        nodes = set([node])
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= neighbors[node] - seen
            yield node
    for node in neighbors:
        if node not in seen:
            yield component(node)


def truth_pairs(data_list, ev, thr):
    data_ev = prepare_test_data(data_list, ev)
    truth_edge_index = data_ev.edge_index
    truth_edge_label = data_ev.edge_label > thr
    truth_nodes_features = data_ev.x
    
    src_edge_index_true = truth_edge_index[0][truth_edge_label]
    dest_edge_index_true = truth_edge_index[1][truth_edge_label]
    edge_scores_Ereco = data_ev.edge_score_Ereco[truth_edge_label]

    index_tuple = []
    for i in range(len(src_edge_index_true)):
        index_tuple.append([src_edge_index_true[i].item(), dest_edge_index_true[i].item()])
    return truth_nodes_features, index_tuple, edge_scores_Ereco


def _get_adj_matrix(X, edge_index, device=None):
    """
    Calculates adjacency matrix and trackster index for the EdgeConv.
    """
    if device is not None:
        device = torch.device(device)
    else:
        use_cuda = torch.cuda.is_available()
        device = torch.device(f'cuda:0' if use_cuda else 'cpu')

    N = X.shape[0]
    E = edge_index.shape[1]
    
    S = torch.zeros(E, N, dtype=float, device=device)
    D = torch.zeros(E, N, dtype=float, device=device)

    for i, (s, d) in enumerate(edge_index.T):
        D[i, d] = 1.
        S[i, s] = 1.

    SRC = torch.cat((torch.eye(N, device=device), S, D), dim=0)
    DST = torch.cat((torch.eye(N, device=device), D, S), dim=0)
    return torch.unsqueeze(SRC, dim=0).float(), torch.unsqueeze(DST, dim=0).float()

def prepare_network_input_data(X, edge_index, edge_features=None, device=None):
    X = torch.nan_to_num(X, nan=0.0)    
    A_unsq, trackster_index_unsq = _get_adj_matrix(X, edge_index, device)
    
    if edge_features is not None:
        edge_features = torch.nan_to_num(edge_features, nan=0.0)
        return torch.unsqueeze(X, dim=0), torch.unsqueeze(edge_index, dim=0).float(), A_unsq, trackster_index_unsq, torch.unsqueeze(edge_features, dim=0).float()
    
    return torch.unsqueeze(X, dim=0), torch.unsqueeze(edge_index, dim=0).float(), A_unsq, trackster_index_unsq


def prediction_pairs(model, data_list, ev, thr, prepare_network_input_data=None, return_net_out=False, device=False, edge_features=False):
    data_ev = prepare_test_data(data_list, ev)
    if prepare_network_input_data is not None:
        if edge_features:
            if data_ev.edge_index.shape[1] != data_ev.edge_features.shape[0]:
                print("ERROR: edge index shape is different from edge features shape")
                return 0
            inputs = prepare_network_input_data(data_ev.x, data_ev.edge_index, data_ev.edge_features,
                                            device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')
            
        else:
            inputs = prepare_network_input_data(data_ev.x, data_ev.edge_index, 
                                            device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    else:
        inputs = (data_ev.x, data_ev.edge_index)
        
    if device:
        out, emb = model(*inputs, device='cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    else: 
        out, emb = model(*inputs)
    edge_index = data_ev.edge_index
    truth_edge_label = out > thr
    node_features = data_ev.x
    
    src_edge_index_true = edge_index[0][truth_edge_label]
    dest_edge_index_true = edge_index[1][truth_edge_label]
    edge_scores_Ereco = data_ev.edge_score_Ereco[truth_edge_label]
    
    index_tuple = []
    for i in range(len(src_edge_index_true)):
        index_tuple.append([src_edge_index_true[i].item(), dest_edge_index_true[i].item()])

    if return_net_out:
        return node_features, index_tuple, edge_scores_Ereco, out
    
    return node_features, index_tuple, edge_scores_Ereco


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        #print("found")
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            #print("found bias")
            m.bias.data.fill_(0.)

def save_pred(pred_flat, lab_flat, epoch=0, out_folder=None):
    
    true_pred = pred_flat[lab_flat==1] 
    false_pred = pred_flat[lab_flat!=1]

    bins = 100
    fig = plt.figure(figsize=(12, 3))
    ax1 = fig.add_subplot(121)
    ax1.hist(false_pred, bins=bins, density=1, label="False predictions", histtype='step')
    ax1.hist(true_pred, bins=bins, density=1, label="True predictions", histtype='step')
    ax1.legend(loc="upper center") #loc="upper left")
    #ax1.set_yscale('log')

    ax2 = fig.add_subplot(122)
    ax2.hist(pred_flat, bins=bins, label="All predictions")
    ax2.legend()

    ax1.set_title("MLP True and False edge prediction distribtion", fontsize=15)
    ax1.set_xlabel("Predicted score", fontsize=14)
    ax1.set_ylabel('Probability [%]', fontsize=14)
    ax2.set_title("MLP edge prediction distribtion", fontsize=15)
    ax2.set_xlabel("Predicted score", fontsize=14)
    ax2.set_ylabel('Counts', fontsize=14)
    plt.show()

    if out_folder is not None:
        fig.savefig(f'{out_folder}/double-pion-0-pu-edge-pred-distributions-epoch-{epoch+1}.pdf', dpi=300, bbox_inches='tight', transparent=True)
        fig.savefig(f'{out_folder}/double-pion-0-pu-edge-pred-distributions-eppoch-{epoch+1}.png', dpi=300, bbox_inches='tight', transparent=True)
