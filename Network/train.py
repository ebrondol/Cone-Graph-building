from utils.dataloading import *
from utils.plotting import *
from utils.scores import *
from utils.utils import *
from network.network import *
import pdb

import os
import numpy as np
from pyrsistent import l
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils as Utils
from tqdm import tqdm

import mplhep as hep
import glob
import argparse
import importlib.util

import numpy as np
from test import *

def get_model_prediction(model, testLoader, prepare_network_input_data=None, 
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
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


def evaluate_clusters(dataset, predictions=None, thr=None, geometric=False, numevents=1000, PU=False):
    """
    Calculates homogenity, completeness, v-measure, adjusted random index and mutual info of the clustering.
    Predictions are the list of lists containing network scores for edges.
    """
    hom, compl, vmeas, randind, mutinfo = [], [], [], [], []
    
    for i in tqdm(range(numevents)):
        event = dataset[i]
        if not geometric:
            components, predicted_cluster_ids = find_connected_components(event, predictions[i], edge_index=None, thr=thr, PU=PU)
        else:
            predicted_cluster_ids = event.candidate_match.cpu().detach().numpy()
            
        t_match = event.best_simTs_match.cpu().detach().numpy()
        
        hom.append(metrics.homogeneity_score(t_match, predicted_cluster_ids))
        compl.append(metrics.completeness_score(t_match, predicted_cluster_ids))
        vmeas.append(metrics.v_measure_score(t_match, predicted_cluster_ids))
        randind.append(metrics.adjusted_rand_score(t_match, predicted_cluster_ids))
        mutinfo.append(metrics.adjusted_mutual_info_score(t_match, predicted_cluster_ids))
    
    return hom, compl, vmeas, randind, mutinfo


def get_best_threshold(TNR, TPR, thresholds, epsilon = 0.02, default=0.65):
    # Find the threshold for which TNR and TPR intersect

    for i in range(len(thresholds)-1):
        if abs(TNR[i] - TPR[i]) < epsilon:
            return round(thresholds[i], 3)
    
        if TNR[i] - TPR[i] < 0 and TNR[i+1] - TPR[i+1] >= epsilon:
            return round(0.5*(thresholds[i] + thresholds[i+1]), 3)
    print("Chosen a default threshold...")
    return default

  
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.4):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets):        
        """Binary focal loss, mean.

        Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
        improvements for alpha.
        :param bce_loss: Binary Cross Entropy loss, a torch tensor.
        :param targets: a torch tensor containing the ground truth, 0s and 1s.
        :param gamma: focal loss power parameter, a float scalar.
        :param alpha: weight of the class indicated by 1, a float scalar.
        """
        bce_loss = F.binary_cross_entropy(predictions, targets)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)  
        # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        return f_loss.mean()
    
def get_unique_run(models_path):
    """
    Prepare the output folder run id.
    """
    previous_runs = os.listdir(models_path)
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    return run_number

def train_val(model, trainLoader, valLoader, optimizer, loss_function, epochs, outputModelPath,
              scheduler=None, with_scores=False, update_every=1, save_checkpoint_every=1):
    """
    Performs training and validation of the network for a required number of epochs.
    """    
    outputModelCheckpointPath = outputModelPath + "/checkpoints/"
    outputLossFunctionPath = outputModelPath + "/loss/"
    outputTrainingPlotsPath = outputModelPath + "/trainingPlots/"
    
    # Create directories for saving the models/checkpoints if not exist
    mkdir_p(outputModelPath)
    mkdir_p(outputModelCheckpointPath)
    mkdir_p(outputLossFunctionPath)
    mkdir_p(outputTrainingPlotsPath)
    
    train_loss_history = []
    val_loss_history = []

    print(f"Model output directory: {outputModelPath}")
    print(f"Saving checkpoints every {save_checkpoint_every} epochs.")
    
    print(">>> Model training started.")
    model.to(device)
    
    for epoch in range(epochs):
        batchloss = []
        train_true_seg, train_pred_seg, scores_seg = [], [], []
        b = 0
        num_samples = len(trainLoader)
        optimizer.zero_grad()
                
        # Training ----------------------------------------------------
        model.train()
        for sample in tqdm(trainLoader, desc=f'Training epoch {epoch+1}'):
            sample.to(device)
            
            num_tracksters, feat_dim = sample.x.shape
            if num_tracksters <= 1:
                continue
        
            inputs = prepare_network_input_data(sample.x, sample.edge_index)
            out, emb = model(*inputs)
            if not with_scores:
                loss = loss_function(out, sample.edge_label.to(torch.float32))
            else:
                loss = loss_function(out, sample.edge_score_Ereco)
            batchloss.append(loss.item())
            loss.backward()
            
            if (b+1) % update_every == 0 or (b+1) == num_samples:
                optimizer.step()
                optimizer.zero_grad()
            
            b += 1
            
            seg_np = sample.edge_label.cpu().numpy()
            scores = sample.edge_score_Ereco.cpu().numpy()
            pred_np = out.detach().cpu().numpy()

            train_true_seg.append(seg_np.reshape(-1))
            train_pred_seg.append(pred_np.reshape(-1))
            scores_seg.append(scores.reshape(-1))

        train_true_cls = np.concatenate(train_true_seg)
        train_pred_cls = np.concatenate(train_pred_seg)
        scores_cls = np.concatenate(scores_seg)
        
        plot_prediction_distribution_standard_and_log(train_pred_cls, train_true_cls,
                                                      epoch=epoch+1, thr = 0.65, scores=scores_cls,
                                                      folder=outputTrainingPlotsPath)

        train_loss = np.mean(batchloss)
        
        if hasattr(model, 'writer'):
            model.writer.add_scalar("Loss/train", train_loss, epoch)
            model.writer.flush()
            
        train_loss_history.append(train_loss)
        # End Training ----------------------------------------------------
            
        # Validation ------------------------------------------------------
        val_true_seg, val_pred_seg = [], []
        
        with torch.set_grad_enabled(False):
            batchloss = []
            model.eval()
            for sample in tqdm(valLoader, desc=f'Validation epoch {epoch+1}'):
                sample.to(device)
                
                num_tracksters, feat_dim = sample.x.shape
                if num_tracksters <= 1:
                    continue
                
                inputs = prepare_network_input_data(sample.x, sample.edge_index)
                out, emb = model(*inputs)
                if not with_scores:
                    val_loss = loss_function(out, sample.edge_label.to(torch.float32))
                else:
                    val_loss = loss_function(out, sample.edge_score_Ereco)
                
                batchloss.append(val_loss.item())
                
                seg_np = sample.edge_label.cpu().numpy()
                pred_np = out.detach().cpu().numpy()

                val_true_seg.append(seg_np.reshape(-1))
                val_pred_seg.append(pred_np.reshape(-1))

        val_true_cls = np.concatenate(val_true_seg)
        val_pred_cls = np.concatenate(val_pred_seg)        
                
        print("Testing model on validation data...")
        TNR, TPR, thresholds = classification_thresholds_plot(val_pred_cls, val_true_cls, threshold_step=0.05,
                                       output_folder=outputTrainingPlotsPath, epoch=epoch+1)
        classification_threshold = get_best_threshold(TNR, TPR, thresholds)
        print(f"Chosen classification threshold is: {classification_threshold}")
        
        plot_prediction_distribution_standard_and_log(val_pred_cls, val_true_cls,
                                                      epoch=epoch+1, thr = classification_threshold,
                                                     folder=outputTrainingPlotsPath, val=True)
        
        test_results = test(val_true_cls, val_pred_cls, classification_threshold=classification_threshold,
                            output_folder=outputTrainingPlotsPath, epoch=epoch+1)

        val_loss = np.mean(batchloss)
        
        if hasattr(model, 'writer'):
            model.writer.add_scalar("Loss/val", val_loss, epoch)
            model.writer.flush()
        val_loss_history.append(val_loss)
        # End Validation ----------------------------------------------------
        
        # Save checkpoint every 'save_checkpoint_every' epochs
        if(epoch != 0 and epoch != epochs-1 and (epoch+1) % save_checkpoint_every == 0):
            print("Saving a model checkpoint.")
            torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss_history,
                        }, outputModelCheckpointPath + f"/epoch_{epoch+1}_val_loss{val_loss:.6f}.pt")
        
        if scheduler is not None:
            scheduler.step(val_loss_history[-1])
            for param_group in optimizer.param_groups:
                print(f"lr: {param_group['lr']}")
            
        print(f"epoch {epoch+1}: Train loss: {train_loss_history[-1]} \t Val loss: {val_loss_history[-1]}")
        
        # Save the updated picture and pkl files of the losses 
        save_loss(train_loss_history, val_loss_history, outputLossFunctionPath)

    # Save the final model
    print(f">>> Training finished. Saving model to {outputModelPath + '/model.pt'}")
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_history,
                }, outputModelPath + "/model.pt")
    
    if hasattr(model, 'writer'):
        model.writer.close()
        
    return train_loss_history, val_loss_history

def save_loss(train_loss_history, val_loss_history, outputLossFunctionPath,title):
    # Saving the figure of the training and validation loss
    fig = plt.figure(figsize=(10, 8))
    #plt.style.use('seaborn-whitegrid')
    epochs = len(train_loss_history)
    plt.plot(range(1, epochs+1), train_loss_history, label='train', linewidth=2)
    plt.plot(range(1, epochs+1), val_loss_history, label='val', linewidth=2)
    plt.ylabel("Loss", fontsize=22)
    plt.xlabel("Epochs", fontsize=22)
    plt.title("Training and Validation Loss", fontsize=24)
    plt.legend()
    plt.savefig(f"{outputLossFunctionPath}/{title}.png")
    plt.show()
    
    # Save the train and validation loss histories to pkl files
    with open(outputLossFunctionPath + 'train_loss_history.pkl','wb') as f:
        pickle.dump(train_loss_history, f)

    with open(outputLossFunctionPath + 'val_loss_history.pkl','wb') as f:
        pickle.dump(val_loss_history, f)

def print_training_dataset_statistics(trainDataset):
    print(f"Number of events in training dataset: {len(trainDataset)}")
    num_nodes, num_edges, num_neg, num_pos = 0, 0, 0, 0
    for ev in trainDataset:
        num_nodes += ev.num_nodes
        num_edges += len(ev.edge_label)
        num_pos += (ev.edge_label == 1).sum()
        num_neg += (ev.edge_label == 0).sum()
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of positive edges: {num_pos}")
    print(f"Number of negative edges: {num_neg}")

def train(model, opt, loader, epoch, device="cuda", edge_features=False,Contrastive = False):

    epoch_loss, epoch_loss_ec, epoch_loss_nc = 0,0,0
    model.train()
    for sample in tqdm(loader, desc=f"Training epoch {epoch}"):
        # reset optimizer and enable training mode
        opt.zero_grad()

        # move data to the device
        sample = sample.to(device)

        # get the prediction tensor
        if edge_features:
            if sample.edge_index.shape[1] != sample.edge_features.shape[0]:
                continue
            data = prepare_network_input_data(sample.x, sample.edge_index, sample.edge_features, device=device)
        else:
            data = prepare_network_input_data(sample.x, sample.edge_index, device=device)
        z, emb, edg_emb, nc_pred, nc_node_em = model(*data)    

        # compute the edge prediction loss
        if Contrastive:  
            edge_loss = loss_obj(edg_emb,sample.edge_label.float())
        else:
            edge_loss = loss_obj(z, sample.edge_label.float())


        # compute node classification loss
        pred_sc_energy = get_regressed_sc_energy(nc_pred,sample.best_simTs_match, sample.sc_energy)
        non_nan_idx = torch.nonzero(~torch.isnan(pred_sc_energy)).squeeze()
        energy_loss = nn.MSELoss()(pred_sc_energy[non_nan_idx],sample.sc_energy[non_nan_idx]/100)

        # print((pred_sc_energy[non_nan_idx].sum()/sample.sc_energy[non_nan_idx].sum()*100))

        loss = edge_loss + energy_loss

        epoch_loss_ec += edge_loss
        epoch_loss_nc += energy_loss
        epoch_loss += loss
        if epoch_loss.isnan():
            pdb.set_trace()

        # back-propagate and update the weight
        loss.backward()
        opt.step()

    return float(epoch_loss), float(epoch_loss_ec), float(epoch_loss_nc)



def train_reg(model, opt, loader, epoch, device="cuda", edge_features=False,reg = None):

    epoch_loss = 0
    model.train()
    for sample in tqdm(loader, desc=f"Training epoch {epoch}"):
        # reset optimizer and enable training mode
        opt.zero_grad()

        # move data to the device
        sample = sample.to(device)

        # get the prediction tensor
        if edge_features:
            if sample.edge_index.shape[1] != sample.edge_features.shape[0]:
                continue
            data = prepare_network_input_data(sample.x, sample.edge_index, sample.edge_features, device=device)
        else:
            data = prepare_network_input_data(sample.x, sample.edge_index, device=device)
        z, emb,_ = model(*data, device)    

        # compute the loss
        if reg == None:
            loss = loss_obj(z, sample.edge_label.float())
        if reg == "L1":
            loss = loss_obj(z, sample.edge_label.float())
            l1_lambda = .001
            l1_norm = sum(torch.linalg.norm(p,1) for p in model.parameters())
            loss = loss +l1_lambda*l1_norm
        if reg == "L2":
            loss = loss_obj(z, sample.edge_label.float())
            l2_lambda = .001
            l2_norm = sum(torch.linalg.norm(p,1) for p in model.parameters())
            loss = loss +l2_lambda*l2_norm
        epoch_loss += loss
        

        # back-propagate and update the weight
        loss.backward()
        opt.step()

    return float(epoch_loss),


from torch.optim.lr_scheduler import CosineAnnealingLR

if __name__ == "__main__":
    # Load the dataset
#E    dataset_location = "/eos/user/a/arouyer/SWAN_projects/closeByDoublePion_dataset_TICL_graph_33_properties"
    dataset_location = "/eos/user/e/ebrondol/SWAN_projects/Cone-Graph-building2/dataproduction/closeByDoublePion_dataset/test_3/"
    #dataset_location = "/afs/cern.ch/user/e/ebrondol/public/4Mark/closeByDoublePion_dataset/vanilla"

    print(">>> Loading datasets...")
    trainDataset = torch.load(f"{dataset_location}/dataTraining.pt")
    valDataset = torch.load(f"{dataset_location}/dataVal.pt")
    testList = torch.load(f"{dataset_location}/dataTest.pt")

#E   testDataset = []
#E
#E   for ev in range(len(testList)):
#E       data = prepare_test_data(testList, ev)
#E       testDataset.append(data)
#E   print(">>> Loaded.")

    # Imbalance in training
    train_edges_total, train_edges_true, train_edges_false, num_nodes = 0, 0, 0, 0
    num_ev =  len(trainDataset)
    print(f"Number of events in training dataset: {num_ev}")

    for tr_data in trainDataset:
        num_nodes += tr_data.num_nodes
        train_edges_total += len(tr_data.edge_index[0])
        train_edges_true += tr_data.edge_label.sum().numpy()
        train_edges_false += (1 - tr_data.edge_label).sum().numpy()
            
    print(f"Number of nodes: {num_nodes}")
    print(f"Total edges in training : {train_edges_total}")
    print(f"True edges : {train_edges_true} - {train_edges_true/train_edges_total*100}%")
    print(f"False edges : {train_edges_false} - {train_edges_false/train_edges_total*100}%")
    print(f"Mean number of edges per event: {train_edges_total/num_ev}")

    alpha = round(1 - train_edges_true/train_edges_total, 3)
    print(f"Setting focal loss alpha to: {alpha}")

    GPU_index = 0
    print(f"Torch version: {torch.__version__}")
    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{GPU_index}' if use_cuda else 'cpu')
    if use_cuda:
        print('>>> CUDNN VERSION:', torch.backends.cudnn.version())
        print('>>> Number CUDA Devices:', torch.cuda.device_count())
        print('>>> CUDA Device Name:', torch.cuda.get_device_name(GPU_index))
        print('>>> CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(GPU_index).total_memory/1e9)
    print(f"Using device: {device}")

    # Create DataLoaders
    train_dl = DataLoader(trainDataset, batch_size=1, shuffle=True)
    val_dl = DataLoader(valDataset, batch_size=1, shuffle=True)

    # Create Model
    model = GNN_TracksterLinkingAndRegressionNet(input_dim = trainDataset[0].x.shape[1], 
                                    edge_feature_dim=0,
                                    edge_hidden_dim=0, hidden_dim=64, weighted_aggr=True,device=device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    #scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    loss_obj = torch.nn.BCELoss()
    #model.apply(weight_init)

    epochs = 5
    decision_th = 0.85
    outputModelPath = "/eos/home-e/ebrondol/SWAN_projects/Cone-Graph-building2/output/test_3/"
    mkdir_p(outputModelPath)

    # Training Loop

    train_loss_hist,train_loss_nc_hist,train_loss_ec_hist = [],[],[]
    val_loss_hist,val_loss_nc_hist,val_loss_ec_hist = [],[],[]
    for epoch in range(epochs):
        loss,loss_ec,loss_nc = train(model, optimizer, train_dl, epoch+1, device, edge_features=False)
#E        loss = train(model, optimizer, train_dl, epoch+1, device, edge_features=True)
        train_loss_hist.append(loss)
        print(f'Epoch: {epoch+1}, train loss: {loss:.4f}, train e.c. loss: {loss_ec:.4f}, train n.c. loss: {loss_nc:.4f}')
        
        print(f">>> Saving model to {outputModelPath + f'/model_epoch_{epoch+1}_loss_{loss:.4f}.pt'}")
        torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, outputModelPath + f'/model_epoch_{epoch+1}_loss_{loss:.4f}.pt')

        # Validation
          
        model.eval()
        pred, lab = [], []
        val_pred_loss, j = 0, 0
        val_energy_loss = 0

        # Define variables for histograms
        rng = [0,2]
        nbins = 50
        nhists = 6
        bs = (rng[1]-rng[0])/nbins
        reg_histos = {"regressed":{},
            "unregressed":{}}
        for i in range(nhists):
            reg_histos["regressed"][i]=np.zeros(nbins)
            reg_histos["unregressed"][i]=np.zeros(nbins)

        for sample in val_dl:
            sample = sample.to(device)
#E            if sample.edge_index.shape[1] != sample.edge_features.shape[0]:
#E                continue
#E            data = prepare_network_input_data(sample.x, sample.edge_index, sample.edge_features, device=device)
            data = prepare_network_input_data(sample.x, sample.edge_index, None, device=device)
            nn_pred,emb, edge_emb,nc_pred, nc_node_em = model(*data)
#E            pred += nn_pred.tolist()
#E            lab += sample.edge_label.tolist()
            val_pred_loss += loss_obj(nn_pred, sample.edge_label.float()).item()

            pred_sc_energy = get_regressed_sc_energy(nc_pred,sample.best_simTs_match, sample.sc_energy)
            non_nan_idx = torch.nonzero(~torch.isnan(pred_sc_energy)).squeeze()
            val_energy_loss += nn.MSELoss()(pred_sc_energy[non_nan_idx],sample.sc_energy[non_nan_idx]/100).detach().cpu()

            # Fill Histograms
            unregressed_sc_energy = get_regressed_sc_energy(sample.x[:,13].unsqueeze(dim=1),sample.best_simTs_match,sample.sc_energy)
            for ele in non_nan_idx:
                eidx = int(sample.sc_energy[ele]/100)
                regressed_frac = pred_sc_energy[ele]/sample.sc_energy[ele]
                idx = int(regressed_frac/bs)
                if idx > 50:
                    continue
                reg_histos["regressed"][eidx][idx] +=1 

                unregressed_frac = unregressed_sc_energy[ele]/sample.sc_energy[ele]
                idx = int(unregressed_frac/bs)
                if idx > 50: # Bug: Something seems to be wrong with the calculation
                    continue
                reg_histos["unregressed"][eidx][idx] +=1

            j += 1
            
        val_energy_loss = float(val_energy_loss)/j
        val_pred_loss = float(val_pred_loss)/j
        val_loss = val_energy_loss + val_pred_loss
        print(f'Epoch: {epoch+1}, val loss: {val_loss:.4f}')
        val_loss_hist.append(val_loss)
        
        #TNR, TPR, thresholds = classification_thresholds_plot(np.array(pred), np.array(lab),
        #                                                    threshold_step=0.05, output_folder=outputModelPath,
        #                                                    epoch=epoch+1)
        #classification_threshold = get_best_threshold(TNR, TPR, thresholds)
        #print(f"Chosen classification threshold is: {classification_threshold}")

        #plot_prediction_distribution_standard_and_log(np.array(pred), np.array(lab),
        #                                            epoch=epoch+1, thr = classification_threshold,
        #                                            folder=outputModelPath, val=True)





        plot_energy_regression_histograms(reg_histos, rng, nbins,folder=outputModelPath,val=True)
        #save_pred(np.array(pred), np.array(lab), epoch=epoch, out_folder=outputModelPath)
        save_loss(train_loss_hist, val_loss_hist, outputLossFunctionPath=outputModelPath,title="losses")
        save_loss(train_loss_nc_hist, val_loss_nc_hist, outputLossFunctionPath=outputModelPath,title="losses_nc")
        save_loss(train_loss_ec_hist, val_loss_ec_hist, outputLossFunctionPath=outputModelPath,title="losses_ec")

        scheduler.step()   

