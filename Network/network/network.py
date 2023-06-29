
# GNN Model 0
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc, f1_score, fbeta_score
from torch.utils.tensorboard import SummaryWriter
import pdb

def _get_graph_feature_from_adj(x, A, trackster_index):
    """
    Create a dynamic graph based on the neigbourhood.
    """        
    out_features = torch.cat((torch.matmul(trackster_index, x), torch.matmul(A-trackster_index, x)), dim=1)
    return out_features

class EdgeConvBlock(nn.Module):
    """EdgeConv layer.
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """

    def __init__(self, in_feat, out_feats, device, activation=True, dropout=0.2, weighted_aggr=False):
        super(EdgeConvBlock, self).__init__()
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = _get_graph_feature_from_adj
        self.weighted_aggr = weighted_aggr
        self.device = device
        
        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Linear(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i]))

        self.acts = nn.ModuleList()
        for i in range(self.num_layers):
            self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Linear(in_feat, out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, features, Adj, trackster_index, alpha=None):

        x = self.get_graph_feature(features, Adj, trackster_index)

        i = 0
        for conv, act in zip(self.convs, self.acts):
            
            x = conv(x)  # (N, C', P, K)
            if self.activation:
                x = act(x)
            if i == 0:
                x = self.drop(x)
            i += 1

        # Do aggregation
        if self.weighted_aggr and alpha is not None:
            N = features.shape[0]
            alpha_vec = torch.cat((torch.ones(N, device=self.device).float(), torch.squeeze(alpha)), dim=0)
            
            # TODO: shape problem
            #print("N ", N)
            #print("x shape ", x.shape)
            #print("alpha shape ", alpha_vec.shape)
            #print("dot product shape ", torch.mul(alpha_vec, x.T).T)
            #print("tr idx transp ", trackster_index.transpose(0, 1))
            x = torch.matmul(trackster_index.transpose(0, 1), torch.mul(alpha_vec, x.T).T)
        else:
            x = torch.matmul(trackster_index.transpose(0, 1), x)
        
        # Skip connection:
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
        else:
            sc = features
        out = self.sc_act(sc + x)  # (N, C_out, P)

        return out
    

def get_model_prediction(model, testLoader, device, prepare_network_input_data=None):
    """
    Gets model predictions on test edges.
    model: the trained network.
    testLoader: DataLoader of already pre-processed data.
    """
    
    model.to(device)
    predictions, truth = [], []
    
    for sample in tqdm(testLoader, desc="Getting model predictions"):
        sample = sample.to(device)
        
        if prepare_network_input_data is not None:
            inputs = prepare_network_input_data(sample.x, sample.edge_index)
        else:
            inputs = (sample.x, sample.edge_index)
            
        link_pred, emb = model(*inputs)
        predictions.append(link_pred.cpu().detach().numpy())
        truth.append(sample.edge_label.cpu().detach().numpy())
    return truth, predictions

def get_model_prediction(model, testLoader, device, prepare_network_input_data=None):
    """
    Gets model predictions on test edges.
    model: the trained network.
    testLoader: DataLoader of already pre-processed data.
    """
    
    model.to(device)
    predictions, truth = [], []
    
    for sample in tqdm(testLoader, desc="Getting model predictions"):
        sample = sample.to(device)
        
        if prepare_network_input_data is not None:
            inputs = prepare_network_input_data(sample.x, sample.edge_index)
        else:
            inputs = (sample.x, sample.edge_index)
            
        link_pred, emb = model(*inputs)
        predictions.append(link_pred.cpu().detach().numpy())
        truth.append(sample.edge_label.cpu().detach().numpy())
    return truth, predictions


class GNN_TracksterLinkingNet_multi(nn.Module):
    # hidden_dim=64
    def __init__(self, device, input_dim=33, hidden_dim=256, output_dim=1, niters=4, dropout=0.2, normalize_stop=30,
                 reconstruction_volume=(3 - 1.5)*2*47*2, both_way_edge_emb=False, edge_feature_dim=0,
                 edge_hidden_dim=0, weighted_aggr=False,multi_head = 0):
        super(GNN_TracksterLinkingNet_multi, self).__init__()
        
        self.writer = SummaryWriter(f"tensorboard_runs/gnn_model_no_geometric")
        
        self.niters = niters
        self.input_dim = input_dim
        self.normalize_stop = normalize_stop
        self.reconstruction_volume = reconstruction_volume
        self.both_way_edge_emb = both_way_edge_emb
        self.edge_feature_dim = edge_feature_dim
        self.weighted_aggr = weighted_aggr
        self.multi_head = multi_head
        self.device = device
        self.ncniters = 1
    
        # Feature transformation to latent space
        self.inputnetwork = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Edge Feature transformation to latent space
        self.edge_inputnetwork = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU()
        )
        
        self.attention_direct = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.attention_reverse = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )
        #multi-head attentions
        self.multi_head_list_dir = nn.ModuleList()
        self.multi_head_list_rev = nn.ModuleList()
        for i in range(multi_head):
            self.multi_head_list_dir.append(nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()))
        
            self.multi_head_list_rev.append(nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        ))
        # EdgeConv
        self.graphconvs = nn.ModuleList()
        for i in range(niters):
            self.graphconvs.append(EdgeConvBlock(in_feat=hidden_dim, 
                                                 out_feats=[2*hidden_dim, hidden_dim],device=self.device, dropout=dropout,
                                                 weighted_aggr=weighted_aggr))

        # Node Classification: Embedding
        #self.ncgraphconvs = nn.ModuleList()
        #for i in range(self.ncniters):
        #    self.ncgraphconvs.append(EdgeConvBlock(in_feat=hidden_dim, 
        #                                        out_feats=[2*hidden_dim, hidden_dim],device=self.device, dropout=dropout,
        #                                        weighted_aggr=weighted_aggr))
        
        # Edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim + edge_hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, X, edge_index, Adj, trackster_index, edge_features=None):
        
        X = torch.squeeze(X, dim=0)
        X = X[:, :self.input_dim]
        X_norm = torch.zeros_like(X)
        
        #print("num edges: ", edge_index.shape)
        #print("num nodes: ", X.shape)
        
        stop = self.normalize_stop
        epsilon = 10e-5 * torch.ones(X[:, :stop].shape, device=self.device)
        std = X[:, :stop].std(dim=0, unbiased=False) + epsilon
        X_norm[:, :stop] = (X[:, :stop] - X[:, :stop].mean(dim=0)) / std
            
        if self.input_dim > self.normalize_stop:
            # Normalizing total number of LCs and trackster -> turning it to density
        
            X_norm[:, stop] = X[:, stop] / (1000 * self.reconstruction_volume)
            X_norm[:, stop+1] = X[:, stop+1] / (100 * self.reconstruction_volume)
            
        if self.input_dim == 33:
            # normalize across existent entries and set the rest to -1
            valid_time_idx = X[:, -1] > -99
            invalid_time_idx = X[:, -1] <= -99
            valid_time_X = X[valid_time_idx]
            
            epsilon_time = 10e-5 * torch.ones(valid_time_X[:, -1].shape, device=self.device)
            std_time = valid_time_X[:, -1].std(dim=0, unbiased=False) + epsilon_time
            
            normalized_time = (valid_time_X[:, -1] - valid_time_X[:, -1].mean(dim=0)) / std_time
            
            X_norm[valid_time_idx, -1] = normalized_time
            X_norm[invalid_time_idx, -1] = -10 
            
        alpha = None
        # Standirdize `edge_features` if present
        if self.edge_feature_dim != 0:
            if edge_features is not None:
                edge_features = torch.squeeze(edge_features, dim=0)
                edge_features_norm = torch.zeros_like(edge_features)
                epsilon = 10e-5 * torch.ones(edge_features.shape, device=self.device)
                std = edge_features.std(dim=0, unbiased=False) + epsilon
                edge_features_norm = (edge_features - edge_features.mean(dim=0)) / std
                # TODO: time normalization should be different!!!
                edge_features_NN = self.edge_inputnetwork(edge_features_norm)
                
                if self.weighted_aggr:
                    #print('edge features NN', edge_features_NN.shape)
                    if self.multi_head !=0:
                        alpha_dir = []
                        alpha_rev = []
                        for dir_heads,rev_heads in zip(self.multi_head_list_dir,self.multi_head_list_rev):
                            alpha_dir.append(dir_heads(edge_features_NN))
                            alpha_rev.append(rev_heads(edge_features_NN))
                        alpha_dir = torch.div(torch.sum(torch.stack(alpha_dir, dim=0),dim=0),self.multi_head)
                        alpha_rev = torch.div(torch.sum(torch.stack(alpha_rev, dim=0),dim=0),self.multi_head)
                        alpha = torch.cat([alpha_dir,alpha_rev],dim = 0)
                        #
                        #
                        #
                    else:
                        alpha_dir = self.attention_direct(edge_features_NN)
                        alpha_rev = self.attention_reverse(edge_features_NN)
                        alpha = torch.cat([alpha_dir, alpha_rev], dim=0)
                
                    #print('alpha', alpha.shape)
            else:
                print("ERROR: Edge features not provided!")
                return 100, 100
        
        
        edge_index = torch.squeeze(edge_index, dim=0).long()
        Adj = torch.squeeze(Adj, dim=0)
        trackster_index = torch.squeeze(trackster_index, dim=0)
        
        # Feature transformation to latent space
        node_emb = self.inputnetwork(X_norm)

        # Niters x EdgeConv block
        for graphconv in self.graphconvs:
            node_emb = graphconv(node_emb, Adj, trackster_index, alpha=alpha)
            
        src, dst = edge_index
        
        if self.edge_feature_dim != 0:
            #print(node_emb[src].shape, node_emb[dst].shape, edge_features_norm.shape)
            edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_features_NN, edge_features_norm], dim=-1)
        else:
            edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            
        pred = self.edgenetwork(edge_emb).squeeze(-1)
        
        if self.both_way_edge_emb:
            edge_emb_reversed = torch.cat([node_emb[dst], node_emb[src]], dim=-1)
            pred_reversed = self.edgenetwork(edge_emb_reversed).squeeze(-1) 
            #pred = (pred+pred_reversed)/2
            pred = torch.min(pred, pred_reversed)


        # Node Classification
        #nc_node_emb = node_emb
        #for graphconv in self.ncgraphconvs:
        #    nc_node_emb = graphconv(nc_node_emb, Adj, trackster_index, alpha=alpha)

        #nc_node_emb = self.node_classifier()
            
        return pred, node_emb,edge_emb



class GNN_TracksterLinkingNet(nn.Module):
    # hidden_dim=64
    def __init__(self, device, input_dim=33, hidden_dim=256, output_dim=1, niters=4, dropout=0.2, normalize_stop=30,
                 reconstruction_volume=(3 - 1.5)*2*47*2, both_way_edge_emb=False, edge_feature_dim=0,
                 edge_hidden_dim=0, weighted_aggr=False):
        super(GNN_TracksterLinkingNet, self).__init__()
        
        self.writer = SummaryWriter(f"tensorboard_runs/gnn_model_no_geometric")
        
        self.niters = niters
        self.input_dim = input_dim
        self.normalize_stop = normalize_stop
        self.reconstruction_volume = reconstruction_volume
        self.both_way_edge_emb = both_way_edge_emb
        self.edge_feature_dim = edge_feature_dim
        self.weighted_aggr = weighted_aggr
        self.device = device
        
        # Feature transformation to latent space
        self.inputnetwork = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Edge Feature transformation to latent space
        self.edge_inputnetwork = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU()
        )
        
        self.attention_direct = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.attention_reverse = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # EdgeConv
        self.graphconvs = nn.ModuleList()
        for i in range(niters):
            self.graphconvs.append(EdgeConvBlock(in_feat=hidden_dim, 
                                                 out_feats=[2*hidden_dim, hidden_dim], device=self.device,dropout=dropout,
                                                 weighted_aggr=weighted_aggr))
        
        self.nc_graphconvs = nn.ModuleList()
        for i in range(niters):
            self.nc_graphconvs.append(EdgeConvBlock(in_feat=hidden_dim, 
                                                    out_feats=[2*hidden_dim, hidden_dim], device=self.device,dropout=dropout,
                                                    weighted_aggr=True))

        # Edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim + edge_hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    
    def forward(self, X, edge_index, Adj, trackster_index, edge_features=None):
        
        X = torch.squeeze(X, dim=0)
        X = X[:, :self.input_dim]
        X_norm = torch.zeros_like(X)
        
        #print("num edges: ", edge_index.shape)
        #print("num nodes: ", X.shape)
        
        stop = self.normalize_stop
        epsilon = 10e-5 * torch.ones(X[:, :stop].shape, device=self.device)
        std = X[:, :stop].std(dim=0, unbiased=False) + epsilon
        X_norm[:, :stop] = (X[:, :stop] - X[:, :stop].mean(dim=0)) / std
            
        if self.input_dim > self.normalize_stop:
            # Normalizing total number of LCs and trackster -> turning it to density
        
            X_norm[:, stop] = X[:, stop] / (1000 * self.reconstruction_volume)
            X_norm[:, stop+1] = X[:, stop+1] / (100 * self.reconstruction_volume)
            
        if self.input_dim == 33:
            # normalize across existent entries and set the rest to -1
            valid_time_idx = X[:, -1] > -99
            invalid_time_idx = X[:, -1] <= -99
            valid_time_X = X[valid_time_idx]
            
            epsilon_time = 10e-5 * torch.ones(valid_time_X[:, -1].shape, device=self.device)
            std_time = valid_time_X[:, -1].std(dim=0, unbiased=False) + epsilon_time
            
            normalized_time = (valid_time_X[:, -1] - valid_time_X[:, -1].mean(dim=0)) / std_time
            
            X_norm[valid_time_idx, -1] = normalized_time
            X_norm[invalid_time_idx, -1] = -10 
            
        alpha = None
        # Standirdize `edge_features` if present
        if self.edge_feature_dim != 0:
            if edge_features is not None:
                edge_features = torch.squeeze(edge_features, dim=0)
                edge_features_norm = torch.zeros_like(edge_features)
                epsilon = 10e-5 * torch.ones(edge_features.shape, device=self.device)
                std = edge_features.std(dim=0, unbiased=False) + epsilon
                edge_features_norm = (edge_features - edge_features.mean(dim=0)) / std
                # TODO: time normalization should be different!!!
                edge_features_NN = self.edge_inputnetwork(edge_features_norm)
                
                if self.weighted_aggr:
                    #print('edge features NN', edge_features_NN.shape)
                    alpha_dir = self.attention_direct(edge_features_NN)
                    alpha_rev = self.attention_reverse(edge_features_NN)
                    alpha = torch.cat([alpha_dir, alpha_rev], dim=0)
                    #print('alpha', alpha.shape)
            else:
                print("ERROR: Edge features not provided!")
                return 100, 100
        
        
        edge_index = torch.squeeze(edge_index, dim=0).long()
        Adj = torch.squeeze(Adj, dim=0)
        trackster_index = torch.squeeze(trackster_index, dim=0)
        
        # Feature transformation to latent space
        node_emb = self.inputnetwork(X_norm)

        # Niters x EdgeConv block
        for graphconv in self.graphconvs:
            node_emb = graphconv(node_emb, Adj, trackster_index, alpha=alpha)
            
        src, dst = edge_index
        
        if self.edge_feature_dim != 0:
            #print(node_emb[src].shape, node_emb[dst].shape, edge_features_norm.shape)
            edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_features_NN, edge_features_norm], dim=-1)
        else:
            edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            
        pred = self.edgenetwork(edge_emb).squeeze(-1)
        
        if self.both_way_edge_emb:
            edge_emb_reversed = torch.cat([node_emb[dst], node_emb[src]], dim=-1)
            pred_reversed = self.edgenetwork(edge_emb_reversed).squeeze(-1) 
            #pred = (pred+pred_reversed)/2
            pred = torch.min(pred, pred_reversed)
            
        return pred, node_emb,edge_emb



class GNN_TracksterLinkingAndRegressionNet(nn.Module):
    # hidden_dim=64
    def __init__(self, device, input_dim=33, hidden_dim=256, output_dim=1, niters=4, dropout=0.2, normalize_stop=30,
                 reconstruction_volume=(3 - 1.5)*2*47*2, both_way_edge_emb=False, edge_feature_dim=0,
                 edge_hidden_dim=0, weighted_aggr=False):
        super(GNN_TracksterLinkingAndRegressionNet, self).__init__()
        
        self.writer = SummaryWriter(f"tensorboard_runs/gnn_model_no_geometric")
        
        self.niters = niters
        self.input_dim = input_dim
        self.normalize_stop = normalize_stop
        self.reconstruction_volume = reconstruction_volume
        self.both_way_edge_emb = both_way_edge_emb
        self.edge_feature_dim = edge_feature_dim
        self.weighted_aggr = weighted_aggr
        self.device = device
        
        # Feature transformation to latent space
        self.inputnetwork = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Edge Feature transformation to latent space
        self.edge_inputnetwork = nn.Sequential(
            nn.Linear(edge_feature_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU()
        )
        
        
        self.attention_direct = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.attention_reverse = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # EdgeConv
        self.graphconvs = nn.ModuleList()
        for i in range(niters):
            self.graphconvs.append(EdgeConvBlock(in_feat=hidden_dim, 
                                                 out_feats=[2*hidden_dim, hidden_dim], device=self.device,dropout=dropout,
                                                 weighted_aggr=weighted_aggr))

        # Edge features from node embeddings for classification
        self.edgenetwork = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feature_dim + edge_hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        # Node Classification EdgeConv
        self.nc_graphconvs = nn.ModuleList()
        for i in range(niters):
            self.nc_graphconvs.append(EdgeConvBlock(in_feat=hidden_dim, 
                                                 out_feats=[2*hidden_dim, hidden_dim], device=self.device,dropout=dropout,
                                                 weighted_aggr=True))
        
        self.ncnetwork = nn.Sequential(
            nn.Linear(hidden_dim + edge_feature_dim + edge_hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
        
    
    def forward(self, X, edge_index, Adj, trackster_index, edge_features=None):
        
        X = torch.squeeze(X, dim=0)
        X = X[:, :self.input_dim]
        X_norm = torch.zeros_like(X)
        
        #print("num edges: ", edge_index.shape)
        #print("num nodes: ", X.shape)
        
        stop = self.normalize_stop
        epsilon = 10e-5 * torch.ones(X[:, :stop].shape, device=self.device)
        std = X[:, :stop].std(dim=0, unbiased=False) + epsilon
        X_norm[:, :stop] = (X[:, :stop] - X[:, :stop].mean(dim=0)) / std
            
        if self.input_dim > self.normalize_stop:
            # Normalizing total number of LCs and trackster -> turning it to density
        
            X_norm[:, stop] = X[:, stop] / (1000 * self.reconstruction_volume)
            X_norm[:, stop+1] = X[:, stop+1] / (100 * self.reconstruction_volume)
            
        if self.input_dim == 33:
            # normalize across existent entries and set the rest to -1
            valid_time_idx = X[:, -1] > -99
            invalid_time_idx = X[:, -1] <= -99
            valid_time_X = X[valid_time_idx]
            
            epsilon_time = 10e-5 * torch.ones(valid_time_X[:, -1].shape, device=self.device)
            std_time = valid_time_X[:, -1].std(dim=0, unbiased=False) + epsilon_time
            
            normalized_time = (valid_time_X[:, -1] - valid_time_X[:, -1].mean(dim=0)) / std_time
            
            X_norm[valid_time_idx, -1] = normalized_time
            X_norm[invalid_time_idx, -1] = -10 
            
        alpha = None
        # Standirdize `edge_features` if present
        if self.edge_feature_dim != 0:
            if edge_features is not None:
                edge_features = torch.squeeze(edge_features, dim=0)
                edge_features_norm = torch.zeros_like(edge_features)
                epsilon = 10e-5 * torch.ones(edge_features.shape, device=self.device)
                std = edge_features.std(dim=0, unbiased=False) + epsilon
                edge_features_norm = (edge_features - edge_features.mean(dim=0)) / std
                # TODO: time normalization should be different!!!
                edge_features_NN = self.edge_inputnetwork(edge_features_norm)
                
                if self.weighted_aggr:
                    #print('edge features NN', edge_features_NN.shape)
                    alpha_dir = self.attention_direct(edge_features_NN)
                    alpha_rev = self.attention_reverse(edge_features_NN)
                    alpha = torch.cat([alpha_dir, alpha_rev], dim=0)
                    #print('alpha', alpha.shape)
            else:
                print("ERROR: Edge features not provided!")
                return 100, 100
        
        
        edge_index = torch.squeeze(edge_index, dim=0).long()
        Adj = torch.squeeze(Adj, dim=0)
        trackster_index = torch.squeeze(trackster_index, dim=0)
        
        # Feature transformation to latent space
        node_emb = self.inputnetwork(X_norm)

        # Niters x EdgeConv block
        for graphconv in self.graphconvs:
            node_emb = graphconv(node_emb, Adj, trackster_index, alpha=alpha)
            
        src, dst = edge_index
        
        if self.edge_feature_dim != 0:
            #print(node_emb[src].shape, node_emb[dst].shape, edge_features_norm.shape)
            edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_features_NN, edge_features_norm], dim=-1)
        else:
            edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            
        pred = self.edgenetwork(edge_emb).squeeze(-1)
        
        if self.both_way_edge_emb:
            edge_emb_reversed = torch.cat([node_emb[dst], node_emb[src]], dim=-1)
            pred_reversed = self.edgenetwork(edge_emb_reversed).squeeze(-1) 
            #pred = (pred+pred_reversed)/2
            pred = torch.min(pred, pred_reversed)
            

        # Node Classification Network
        nc_node_emb = node_emb
        for graphconv in self.nc_graphconvs:
            nc_node_emb = graphconv(nc_node_emb, Adj, trackster_index, alpha=torch.cat((pred,pred),dim=0))
        
        nc_pred = self.ncnetwork(nc_node_emb)

        return pred, node_emb,edge_emb, nc_pred, nc_node_emb






# def prepare_network_input_data(X, edge_index, edge_features=None, device=None):
#     X = torch.nan_to_num(X, nan=0.0)
#     if edge_features is not None:
#         edge_features = torch.nan_to_num(edge_features, nan=0.0)
#         return torch.unsqueeze(X, dim=0), torch.unsqueeze(edge_index, dim=0).float(), torch.unsqueeze(edge_features, dim=0).float()
#     return torch.unsqueeze(X, dim=0), torch.unsqueeze(edge_index, dim=0).float()
    
# class GNN_TracksterLinkingNet(nn.Module):
#     def __init__(self, input_dim=33, hidden_dim=256, output_dim=1, dropout=0.2, normalize_stop=30,
#                  reconstruction_volume=(3 - 1.5)*2*47*2, both_way_edge_emb=False, edge_feature_dim=0):
#         super(GNN_TracksterLinkingNet, self).__init__()
        
#         self.input_dim = input_dim
#         self.normalize_stop = normalize_stop
#         self.reconstruction_volume = reconstruction_volume
#         self.both_way_edge_emb = both_way_edge_emb
#         self.edge_feature_dim = edge_feature_dim
        
#         # Feature transformation to latent space
#         self.inputnetwork = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(), #nn.ReLU()
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU()
#         )
                
#         # Edge features from node embeddings for classification
#         self.edgenetwork = nn.Sequential(
#             nn.Linear(2*hidden_dim + edge_feature_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim//2),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim//2, output_dim),
#             nn.Dropout(p=dropout),
#             nn.Sigmoid()
#         )
        
    
#     def forward(self, X, edge_index, edge_features=None, device="cuda"):
        
#         X = torch.squeeze(X, dim=0)
#         X = X[:, :self.input_dim]
#         X_norm = torch.zeros_like(X)
        
#         stop = self.normalize_stop
#         epsilon = 10e-5 * torch.ones(X[:, :stop].shape, device=device)
#         std = X[:, :stop].std(dim=0, unbiased=False) + epsilon
#         X_norm[:, :stop] = (X[:, :stop] - X[:, :stop].mean(dim=0)) / std
            
#         if self.input_dim > self.normalize_stop:
#             # Normalizing total number of LCs and trackster -> turning it to density
        
#             X_norm[:, stop] = X[:, stop] / (1000 * self.reconstruction_volume)
#             X_norm[:, stop+1] = X[:, stop+1] / (100 * self.reconstruction_volume)
            
#         if self.input_dim == 33:
#             # normalize across existent entries and set the rest to -1
#             valid_time_idx = X[:, -1] > -99
#             invalid_time_idx = X[:, -1] <= -99
#             valid_time_X = X[valid_time_idx]
            
#             epsilon_time = 10e-5 * torch.ones(valid_time_X[:, -1].shape, device=device)
#             std_time = valid_time_X[:, -1].std(dim=0, unbiased=False) + epsilon_time
            
#             normalized_time = (valid_time_X[:, -1] - valid_time_X[:, -1].mean(dim=0)) / std_time
            
#             X_norm[valid_time_idx, -1] = normalized_time
#             X_norm[invalid_time_idx, -1] = -10 
            
#         # Standirdize `edge_features` if present
#         if self.edge_feature_dim != 0:
#             if edge_features is not None:
#                 edge_features = torch.squeeze(edge_features, dim=0)
#                 edge_features_norm = torch.zeros_like(edge_features)
#                 epsilon = 10e-5 * torch.ones(edge_features.shape, device=device)
#                 std = edge_features.std(dim=0, unbiased=False) + epsilon
#                 edge_features_norm = (edge_features - edge_features.mean(dim=0)) / std
#                 # TODO: time normalization should be different!!!
#             else:
#                 print("ERROR: Edge features not provided!")
#                 return 100, 100
           
            
#         # Feature transformation to latent space
#         node_emb = self.inputnetwork(X_norm)
        
#         src, dst = torch.squeeze(edge_index, dim=0).long()
        
        
#         if self.edge_feature_dim != 0:
#             #print(node_emb[src].shape, node_emb[dst].shape, edge_features_norm.shape)
#             edge_emb = torch.cat([node_emb[src], node_emb[dst], edge_features_norm], dim=-1)
#         else:
#             edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
            
#         pred = self.edgenetwork(edge_emb).squeeze(-1)
        
#         if self.both_way_edge_emb:
#             edge_emb_reversed = torch.cat([node_emb[dst], node_emb[src]], dim=-1)
#             pred_reversed = self.edgenetwork(edge_emb_reversed).squeeze(-1) 
#             #pred = (pred+pred_reversed)/2
#             pred = torch.min(pred, pred_reversed)
        
#         return pred, node_emb