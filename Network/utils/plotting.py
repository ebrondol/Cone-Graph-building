from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import math

def plot_pred_distr(lab, pred, decision_th = 0.6):

    #pred = pred.cpu().detach().numpy()
    #targets = targets.cpu().detach().numpy()

    lab = np.array(lab)
    fig = plt.figure(figsize=(14, 4))
    fig.suptitle('Distributions of the labels', size=18)

    ax1 = fig.add_subplot(141)
    ax1.set_title('Prediction distribution')
    ax1.hist(pred, bins=20)

    ax2 = fig.add_subplot(142)
    ax2.hist(lab, bins=20)
    ax2.set_title('True Edge Labels')
    thresholded = (np.array(pred) > decision_th).astype(int)

    ax3 = fig.add_subplot(143)
    ax3.set_title(f'Predicted Edge Labels for thr: {decision_th}')
    ax3.hist(thresholded, bins=20)
    
    correct_pos = thresholded == lab
    correct_pos &= thresholded == 1
    correct_pos = sum(correct_pos)
    
    correct_neg = thresholded == lab
    correct_neg &= thresholded == 0
    correct_neg = sum(correct_neg)
    
    incorrect_pos = thresholded != lab
    incorrect_pos &= thresholded == 1
    incorrect_pos = sum(incorrect_pos)
    
    incorrect_neg = thresholded != lab
    incorrect_neg &= thresholded == 0
    incorrect_neg = sum(incorrect_neg)
    
    ax4 = fig.add_subplot(144)
    
    width = 0.1
    x = np.arange(2)
    ax4.bar(x-0.05, [correct_neg, incorrect_neg], width, label='Negative')
    ax4.bar(x+0.05, [incorrect_pos, correct_pos], width, label='Positive')
    ax4.legend()

    print(f"Edge labels: number of positive: {lab.sum()}")
    print(f"Predictions: number of positive: {thresholded.sum()}")

    plt.tight_layout()
    plt.show()
    
    
def plot_roc(lab, pred):
    fpr, tpr, _ = roc_curve(lab, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()
    
def plot_tsne(event, predicted_cluster_ids, embs, energy, feature_type='', energy_factor=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    unique_supertr = np.array(np.unique(predicted_cluster_ids), dtype=np.int32)
    supertr_idxs = [np.where(predicted_cluster_ids == idx) for idx in unique_supertr]

    dict_emb = {unigue_idx : (embs[supertr_idx], energy[supertr_idx]) for unigue_idx, supertr_idx in zip(unique_supertr, supertr_idxs)}


    for key, (e, en) in dict_emb.items():
        if key == -1:
            ax1.scatter(e[:, 0], e[:, 1], s=energy_factor*en, alpha=0.5, label=key) # edgecolors='k')
        else:
            ax1.scatter(e[:, 0], e[:, 1], s=energy_factor*en, alpha=0.8, label=key) # edgecolors='k')

    ax1.set_title(f"T-SNE visualization of {feature_type} ({embs.shape[0]} tracksters).\nPredicted SuperTracksters ({unique_supertr.shape[0]})", size=18)

    ####################################

    unique_supertr = np.array(event.best_simTs_match.cpu().unique(), dtype=np.int32)
    supertr_idxs = [np.where(event.best_simTs_match.cpu() == idx) for idx in unique_supertr]

    dict_emb = {unigue_idx : (embs[supertr_idx], energy[supertr_idx]) for unigue_idx, supertr_idx in zip(unique_supertr, supertr_idxs)}


    for key, (e, en) in dict_emb.items():
        if key == -1:
            ax2.scatter(e[:, 0], e[:, 1], s=energy_factor*en, alpha=0.5, label=key) # edgecolors='k')
        else:
            ax2.scatter(e[:, 0], e[:, 1], s=energy_factor*en, alpha=0.8, label=key) #, edgecolors='k')

    #plt.legend(bbox_to_anchor=(1, 0.5), loc='upper left')

    ax2.set_title(f"T-SNE visualization of {feature_type} ({embs.shape[0]} tracksters).\nTrue SuperTracksters ({unique_supertr.shape[0]})", size=18)

    plt.show()


    from matplotlib.lines import Line2D   

def plotTrackster2D(fig, ax2, y, z, match, plot_missed=True, plot_only_true=False, energy=None,
                    indexes=None, edges=None, edges_t=None, proj=None, plot_all_initial_edges=False, all_edges=None):
    fs = 16
    
    if not plot_only_true and not plot_all_initial_edges:
        for ind in edges:
            if(ind not in edges_t):
                if len(ind) == 0:
                    continue
                idx0, idx1 = ind[0], ind[1]
                ax2.plot([y[idx0], y[idx1]], [z[idx0], z[idx1]], 'red', lw = 0.5, label = "Wrong Edges", zorder=0, alpha=0.5)
            else:
                if len(ind) == 0:
                    continue
                idx0, idx1 = ind[0], ind[1]
                ax2.plot([y[idx0], y[idx1]], [z[idx0], z[idx1]], 'black', lw = 0.5, label = "Correct Edges", zorder=0, alpha=0.5)
        if plot_missed:
            for t_ind in edges_t:
                if(t_ind not in edges):
                    if len(t_ind) == 0:
                        continue
                    idx0, idx1 = t_ind[0], t_ind[1]
                    ax2.plot([y[idx0], y[idx1]], [z[idx0], z[idx1]], 'blue', lw = 0.5, label = "Missed Edges", zorder=0, alpha=0.5)
    elif plot_all_initial_edges:
        for t_ind in all_edges:
            if len(t_ind) == 0:
                continue
            idx0, idx1 = t_ind[0], t_ind[1]
            ax2.plot([y[idx0], y[idx1]], [z[idx0], z[idx1]], 'black', lw = 0.5, label = "Initial Edges", zorder=0, alpha=0.5)
    else:
        for t_ind in edges_t:
            if len(t_ind) == 0:
                continue
            idx0, idx1 = t_ind[0], t_ind[1]
            ax2.plot([y[idx0], y[idx1]], [z[idx0], z[idx1]], 'black', lw = 0.5, label = "True Edges", zorder=0, alpha=0.5)
    
        
            
    simTr = np.unique(match)
    colors = cm.rainbow(np.linspace(0, 1, len(simTr)))

    for y_t, z_t, e_t, m_t in zip(y, z, energy, match):
        ax2.scatter(y_t, z_t, s=e_t*10, alpha=0.7, zorder=1, c=colors[int(m_t)].reshape(1,-1), edgecolors='k', linewidth=1)

    ax2.tick_params(axis='x', labelsize=fs)
    ax2.tick_params(axis='y', labelsize=fs)
    ax2.set_xlabel(proj[0] + " (cm)", fontsize = fs)
    ax2.set_ylabel(proj[1] + " (cm)", fontsize = fs) 

def plotTrackster3D(fig, ax, x, y, z, match, plot_missed=True, plot_only_true=False, energy=None, indexes=None, edges=None, edges_t = None,
                    label='Trackster Energy (GeV)', plot_all_initial_edges=False, all_edges=None):
    fs = 15
    missed, correct, wrong = 0, 0, 0
    
    ax.set_xlabel('Z (cm)', fontsize = fs)
    ax.set_ylabel('X (cm)', fontsize = fs)
    ax.set_zlabel('Y (cm)', fontsize = fs)
    
    if not plot_only_true and not plot_all_initial_edges:

        for ind in edges:

            if len(ind) == 0: continue
            idx0, idx1 = ind[0], ind[1]

            if(ind not in edges_t):
                wrong += 1
                ax.plot([x[idx0], x[idx1]], [y[idx0], y[idx1]], [z[idx0], z[idx1]],
                    'red', lw = 0.5, label = "Wrong Edges", alpha=0.5)
            else:
                correct += 1
                ax.plot([x[idx0], x[idx1]], [y[idx0], y[idx1]], [z[idx0], z[idx1]],
                    'black', lw = 0.5, label = "Correct Edges", alpha=0.5)

        if plot_missed:
            for t_ind in edges_t:
                if(t_ind not in edges):
                    missed += 1

                    if len(t_ind) == 0: continue
                    idx0, idx1 = t_ind[0], t_ind[1]

                    ax.plot([x[idx0], x[idx1]], [y[idx0] ,y[idx1]], [z[idx0] ,z[idx1]],
                        'blue', lw = 0.5, label = "Missed Edges", alpha=0.5)
    elif plot_all_initial_edges:
        for t_ind in all_edges:
            
            if len(t_ind) == 0: continue
            idx0, idx1 = t_ind[0], t_ind[1]
            
            ax.plot([x[idx0] ,x[idx1]], [y[idx0] ,y[idx1]], [z[idx0] ,z[idx1]],
                'black', lw = 0.5, label = "Initial Edges", alpha=0.7)
    else:
        for t_ind in edges_t:
            
            if len(t_ind) == 0: continue
            idx0, idx1 = t_ind[0], t_ind[1]
            
            ax.plot([x[idx0] ,x[idx1]], [y[idx0] ,y[idx1]], [z[idx0] ,z[idx1]],
                'black', lw = 0.5, label = "True Edges", alpha=0.7)
            
    simTr = np.unique(match)
    colors = cm.rainbow(np.linspace(0, 1, len(simTr)))

    for x_t, y_t, z_t, e_t, m_t in zip(x, y, z, energy, match):
        ax.scatter(x_t, y_t, z_t, s=e_t*10, alpha=0.7, zorder=1, edgecolors='k', linewidth=1, c=colors[int(m_t)].reshape(1,-1))
        
        
    if not plot_only_true and plot_missed and not plot_all_initial_edges:
        print(f"Edges:\tCorrect num: {correct}, Wrong num: {wrong}, Missed num: {missed}")
    elif plot_all_initial_edges:
        init_edge_num = len(all_edges)
        pred_edge_num = len(edges)
        print(f"Edges:\tInitial num: {init_edge_num} Removed by NN: {init_edge_num - pred_edge_num}")
    else:
        print(f"Edges:\tTrue num: {len(edges_t)}")
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    ax.tick_params(axis='z', labelsize=fs)

def plotTrackster(x, y, z, t_match, predicted_match=None, plot_missed=True,
                  plot_only_true=False, energy=None, indexes=None, edges=None, edges_t=None, plot_all_initial_edges=False, all_edges=None):
    
    if predicted_match is None or plot_only_true or plot_all_initial_edges:
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
    else:
        fig = plt.figure(figsize=(30,25)) 
        ax = fig.add_subplot(441, projection='3d')
        ax2 = fig.add_subplot(442)
        ax3 = fig.add_subplot(443)
        ax4 = fig.add_subplot(444)       
    
    plotTrackster3D(fig, ax, x, y, z, t_match, 
                    plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t, 
                    plot_all_initial_edges=plot_all_initial_edges, all_edges=all_edges)
    ## XY
    plotTrackster2D(fig, ax2, y, z, t_match, 
                    plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t, proj=["X", "Y"],
                    plot_all_initial_edges=plot_all_initial_edges,  all_edges=all_edges)
    ## XZ
    plotTrackster2D(fig, ax3, y, x, t_match, 
                    plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t, proj=["X", "Z"], 
                    plot_all_initial_edges=plot_all_initial_edges, all_edges=all_edges)
    ## YZ
    plotTrackster2D(fig, ax4, z, x, t_match, 
                    plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t, proj=["Y", "Z"],
                    plot_all_initial_edges=plot_all_initial_edges,  all_edges=all_edges)

    # Add legend lines
    if not plot_only_true and not plot_all_initial_edges:
        custom_lines = [Line2D([0], [0], color='red', lw = 4), Line2D([0], [0], color='black', lw=4),
                       Line2D([0], [0], color='blue', lw=4)]
        ax.legend(custom_lines, ['Wrong Edges', 'Correct Edges', "Missed Edges"], fontsize=15)
    elif plot_all_initial_edges:
        custom_lines = [Line2D([0], [0], color='black', lw=4)]
        ax.legend(custom_lines, ['Initial Edges'], fontsize=15)
    else:
        custom_lines = [Line2D([0], [0], color='black', lw=4)]
        ax.legend(custom_lines, ['True Edges'], fontsize=15)
            
    if predicted_match is not None and not plot_only_true and not plot_all_initial_edges:
        ax = fig.add_subplot(445, projection='3d')
        ax2 = fig.add_subplot(446)
        ax3 = fig.add_subplot(447)
        ax4 = fig.add_subplot(448)
        
        plotTrackster3D(fig, ax, x, y, z, predicted_match, 
                        plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t)
        ## XY
        plotTrackster2D(fig, ax2, y, z, predicted_match, 
                        plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t, proj=["X", "Y"])
        ## XZ
        plotTrackster2D(fig, ax3, y, x, predicted_match, 
                        plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t, proj=["X", "Z"])
        ## YZ
        plotTrackster2D(fig, ax4, z, x, predicted_match, 
                        plot_missed, plot_only_true, energy=energy, edges=edges, edges_t=edges_t, proj=["Y", "Z"])

        # Add legend lines
        if not plot_only_true:
            custom_lines = [Line2D([0], [0], color='red', lw = 4), Line2D([0], [0], color='black', lw=4),
                           Line2D([0], [0], color='blue', lw=4)]
            ax.legend(custom_lines, ['Wrong Edges', 'Correct Edges', "Missed Edges"], fontsize=15)
        else:
            custom_lines = [Line2D([0], [0], color='black', lw=4)]
            ax.legend(custom_lines, ['True Edges'], fontsize=15)
        
    plt.tight_layout()
    plt.show()



def classification_thresholds_plot(scores, ground_truth, threshold_step=0.05, output_folder=None, epoch=None):
    """
    Plots and saves the figure of the dependancy of th eaccuracy, True Positive rate (TPR) and 
    True Negative rate (TNR) on the value of the classification threshold.
    """    
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    ACC, TNR, TPR, F1 = [], [], [], []
    for threshold in thresholds:
        
        prediction = scores > threshold
        
        TN, FP, FN, TP = confusion_matrix(ground_truth, prediction).ravel()
        ACC.append((TP+TN)/(TN + FP + FN + TP))
        TNR.append(TN/(TN+FP))
        TPR.append(TP/(TP+FN))
        F1.append(f1_score(ground_truth, prediction))
    
    # Saving the figure of the classification threshold plot
    fig = plt.figure(figsize=(10,8))
    plt.plot(thresholds, ACC, 'go-', label='ACC', linewidth=2)
    plt.plot(thresholds, TNR, 'bo-', label='TNR', linewidth=2)
    plt.plot(thresholds, TPR, 'ro-', label='TPR', linewidth=2)
    plt.plot(thresholds, F1, 'mo-', label='F1', linewidth=2)
    plt.xlabel("Threshold", fontsize=18)
    plt.title("Accuracy / TPR / TNR / F1 based on the classification threshold value", fontsize=18)
    plt.legend()
    
    if output_folder is not None:
        f_name = f"classification_thresholds_epoch_{epoch}.png" if epoch is not None else "classification_thresholds.png"
        plt.savefig(f"{output_folder}/{f_name}")
        plt.show()
       
    return TNR, TPR, thresholds



def plotLoss(model_base_folder):
    plt.style.use(hep.style.CMS)
    #plt.style.use('seaborn-whitegrid')
    train_loss_path = model_base_folder + "/loss/train_loss_history.pkl"
    val_loss_path = model_base_folder + "/loss/val_loss_history.pkl"
    
    with open(train_loss_path, 'rb') as f:
        train_loss = pickle.load(f)
    with open(val_loss_path, 'rb') as f:
        val_loss = pickle.load(f)
        
    fig = plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-whitegrid')
    plt.plot(range(1, len(train_loss)+1), train_loss, label='train', linewidth=2)
    plt.plot(range(1, len(val_loss)+1), val_loss, label='val', linewidth=2)
    plt.ylabel("Loss", fontsize=22)
    plt.xlabel("Epochs", fontsize=22)
    plt.title("Training and Validation Loss", fontsize=24)
    plt.legend()
    plt.show()
    #plt.savefig(f"{model_base_folder}/loss/losses.png")


def plot_prediction_distribution(model, test_dl, threshold=0.7):
    for sample in test_dl:
        sample = sample.to(device)
        pred, emb = model(sample.x, sample.edge_index)
        targets = sample.edge_label
        break
    
    pred = pred.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    fig, axs = plt.subplots(3, figsize=(5, 8))
    fig.suptitle('Distributions of the labels')
    axs[0].set_title('Prediction distribution')
    axs[0].hist(pred, bins=20)
    axs[1].hist(targets, bins=20)
    axs[1].set_title('True Edge Labels')
    thresholded = (np.array(pred) > thr).astype(int)

    axs[2].set_title(f'Predicted Edge Labels for thr: {thr}')
    axs[2].hist(thresholded, bins=20)

    print(f"Edge labels: number of positive: {targets.sum()}")
    print(f"Predictions: number of positive: {thresholded.sum()}")

    plt.tight_layout()
    # hep.cms.text('Preliminary')
    plt.show()

def get_truth_labels(data_list, ev):
    """list of indices of best matched simts to all ts in an event"""
    x_best_simts = data_list[ev][5]
    return x_best_simts
 
def get_cand_labels(data_list, ev):
    """candidates containing the trackster"""
    cand_match = data_list[ev][6]
    return cand_match

def plot_prediction_distribution_standard_and_log(pred, targets, epoch=None, thr = 0.65, scores=None, folder=None, val=False):
    
    fig = plt.figure(figsize=(10,9))
    #plt.style.use('seaborn-whitegrid')

    fig.suptitle(f'Distributions of the labels (Epoch: {epoch})')
    
    num_plots = 4 if scores is not None else 3
    
    # Predictions in linear and log scales
    ax1 = fig.add_subplot(num_plots,2,1)
    ax1.set_title('Prediction distribution')
    ax1.hist(pred, bins=30)
    
    ax2 = fig.add_subplot(num_plots,2,2)
    ax2.set_title('Prediction distribution Log')
    ax2.hist(pred, bins=30)
    ax2.set_yscale('log')
    #------------------------
    
    # Truth labels in linear and log scales
    ax3 = fig.add_subplot(num_plots,2,3)
    ax3.hist(targets, bins=30)
    ax3.set_title('True Edge Labels')
    thresholded = (np.array(pred) > thr).astype(int)
    
    ax4 = fig.add_subplot(num_plots,2,4)
    ax4.hist(targets, bins=30)
    ax4.set_title('True Edge Labels Log')
    ax4.set_yscale('log')
    #------------------------

    # Thresholded labels in linear and log scales
    ax5 = fig.add_subplot(num_plots,2,5)
    ax5.set_title(f'Predicted Edge Labels for thr: {thr}')
    ax5.hist(thresholded, bins=30)
    
    ax6 = fig.add_subplot(num_plots,2,6)
    ax6.set_title(f'Predicted Edge Labels Log for thr: {thr}')
    ax6.hist(thresholded, bins=30)
    ax6.set_yscale('log')
    #------------------------
    
    if scores is not None:
        ax7 = fig.add_subplot(num_plots,2,7)
        ax7.set_title('Edge Scores')
        ax7.hist(scores, bins=30)
        
        ax8 = fig.add_subplot(num_plots,2,8)
        ax8.set_title('Edge Scores Log')
        ax8.hist(scores, bins=30)
        ax8.set_yscale('log')

    print(f"Edge labels: number of positive: {targets.sum()}")
    print(f"Predictions: number of positive: {thresholded.sum()}")

    plt.xlim([0, 1])
    plt.tight_layout()
    if folder is not None:
        name = f"plot_train_prediction_distribution_epoch_{epoch}.png" if val else f"plot_val_prediction_distribution_epoch_{epoch}.png"
        plt.savefig(f'{folder}/{name}', bbox_inches='tight')
    plt.show()



def plot_energy_regression_histograms(histos, rng, nbins, epoch, folder, val=False):
    # The function takes as input a dictionary of histograms, the range for the x axis, and the number of bins
    # for the energy fractions, regressed and unregressed, and for 6 different energy ranges [0,100,200,300,400,500,600]
    
    x = np.linspace(rng[0],rng[1],nbins)
    bs = x[1]-x[0]
    rows = math.ceil(len(histos["unregressed"].keys())/3)
    fig, axs = plt.subplots(rows,3,figsize=(15,15))
    i,j = 0,0 
    for energy in histos["unregressed"].keys():
        axs[i,j].step(x,histos["unregressed"][energy],label="unregressed")
        axs[i,j].step(x,histos["regressed"][energy],label="regressed")
        axs[i,j].grid()
        axs[i,j].legend()
        axs[i,j].set_title("%s-%s GeV"%(str(energy*100),str((energy+1)*100)))
        axs[i,j].set_xlabel(r"$en_{TRKs}/en_{SC}$")
        j=j+1
        if j%3==0:
            i=i+1
            j=0
    fig.suptitle("Energy Regression Histograms")

    plt.tight_layout()
    if folder is not None:
        name = f"plot_train_energy_regression_histogram_epoch_{epoch}.png" if val else f"plot_val_energy_regression_histogram_epoch_{epoch}.png"
        plt.savefig(f'{folder}/{name}', bbox_inches='tight')
    plt.show()


