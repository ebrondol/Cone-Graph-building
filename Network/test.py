def test(truth, scores, classification_threshold = 0.7, output_folder=None, epoch=None):
    
    results = {}
    print(scores)
    predictions = scores > classification_threshold    
    sample_weight = compute_sample_weight(class_weight='balanced', y=truth)
    
    cf_matrix = confusion_matrix(truth, predictions)
    print(f"Confusion matrix:\n{cf_matrix}\n")
    
    cf_matrix_w_norm = confusion_matrix(truth, predictions, sample_weight=sample_weight, normalize='all')
    print(f"Confusion matrix weighted:\n{cf_matrix_w_norm}\n")
    
    TN, FP, FN, TP = cf_matrix.ravel()
    print(f"TN: {TN} \t FN: {FN} \t TP: {TP} \t FP: {FP}")
    results["F1"] = f1_score(truth, predictions)
    results["BA"] = balanced_accuracy_score(truth, predictions)
    
    # Sensitivity, hit rate, recall, or true positive rate
    results["TPR"] = TP/(TP+FN)
    # Specificity or true negative rate
    results["TNR"] = TN/(TN+FP) 
    # Precision or positive predictive value
    results["PPV"] = TP/(TP+FP)
    # Negative predictive value
    results["NPV"] = TN/(TN+FN)
    # Fall out or false positive rate
    results["FPR"] = FP/(FP+TN)
    # False negative rate
    results["FNR"] = FN/(TP+FN)
    # False discovery rate
    results["FDR"] = FP/(TP+FP)
    
    tot = TN + FP + FN + TP
    ACC = (TP+TN)/tot
    # normalized to total edges in test dataset
    print(f"Confusion matrix scaled:\n{cf_matrix/tot}\n")
    print(f"Accuracy: {ACC:.4f}")
    print(f"Precision: {results['PPV']:.4f}")
    print(f"Negative predictive value: {results['NPV']:.4f}")
    print(f"Recall: Correctly classifying {results['TPR']*100:.4f} % of positive edges")
    print(f"True negative rate: Correctly classifying {results['TNR']*100:.4f} % of all negative edges")
    print(f"F1 score: {results['F1']:.4f}")
    
    prec_w, rec_w, fscore_w, _ = precision_recall_fscore_support(truth, predictions, sample_weight=sample_weight)
    print(prec_w, rec_w, fscore_w)
    print(f"Balanced accuracy: {results['BA']:.4f}")
    print(f"Precision weighted: {prec_w}")
    print(f"Recall weighted: {rec_w}")
    print(f"F1 score weighted: {fscore_w}")
    
    # computes the positive and negative likelihood ratios (LR+, LR-) to assess the predictive 
    # power of a binary classifier. As we will see, these metrics are independent of the proportion between
    #classes in the test set, which makes
    #them very useful when the available data for a study has a different class proportion than the target application.
    pos_lr, neg_lr = class_likelihood_ratios(truth, predictions, raise_warning=False)
    print(f"positive_likelihood_ratio: {pos_lr}, negative_likelihood_ratio: {neg_lr}")

    max_el = max(np.amax(cf_matrix/tot), np.amax(cf_matrix_w_norm))
    
    # plot confusion matrix
    fig, px = plt.subplots(1, 2, figsize=(8, 4))
    plt.set_cmap("viridis")
    px[0].set_xlabel("Predicted")
    px[0].set_ylabel("True")
    cax = px[0].matshow(cf_matrix/tot)
    
    px[0].set_title(f"(ACC: {ACC:.4f}, TPR: {results['TPR']:.4f}, TNR: {results['TNR']:.4f})\nThreshold: {classification_threshold}",
              fontsize=14)

    px[1].set_xlabel("Predicted")
    px[1].set_ylabel("True")
    cax = px[1].matshow(cf_matrix_w_norm)
    px[1].set_title(f"(BA: {results['BA']:.4f}, TPR: {results['TPR']:.4f}, TNR: {results['TNR']:.4f})\n Threshold: {classification_threshold}",
              fontsize=14)
    #fig.colorbar(cax)
    
    results["TN"], results["FP"], results["FN"], results["TP"] = TN, FP, FN, TP
    results["ACC"] = ACC
    results["cf_matrix"] = cf_matrix
    # Scores output by the neural network without classification
    results["scores"] = scores
    results["prediction"] = predictions
    results["ground_truth"] = truth
    results["classification_threshold"] = classification_threshold
    
    if output_folder is not None:
        f_name = f"confusion_matrix_epoch_{epoch}.png" if epoch is not None else "confusion_matrix_epoch.png"
        plt.savefig(f"{output_folder}/{f_name}")
        
        pkl_name = f"test_results_{epoch}.pkl" if epoch is not None else "test_results.pkl"
        with open(f'{output_folder}/{pkl_name}','wb') as f:
            pickle.dump(results, f)
    plt.show()
    #----------------------------------
            
    fpr, tpr, _ = roc_curve(truth, scores)
    results["ROC_AUC"] = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % results["ROC_AUC"])
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title("Receiver operating characteristic", fontsize=24)
    plt.legend(loc="lower right")
    
    if output_folder is not None:
        roc_name = f"roc_{epoch}.png" if epoch is not None else "roc.png"
        plt.savefig(f"{output_folder}/{roc_name}")
    plt.show()
            
    return results   


def load_model_for_testing(modelFolder):
    
    modelPath = f"{modelFolder}/model.pt"
    model_architecture_file = f"{modelFolder}/architecture.py"
    
    # Loading the model file
    print(">>> Loading model from the provided path...")
    spec = importlib.util.spec_from_file_location("model", model_architecture_file)
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)
    model = model_lib.TracksterLinkingNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(modelPath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model epoch: {checkpoint['epoch']}")
    print(">>> Model Loaded.")
    prepare_network_input_data = None
    try:
        prepare_network_input_data = model_lib.prepare_network_input_data
    except Exception as ex:
        print(ex)
    return model, prepare_network_input_data


@torch.no_grad()
def test_simple(model, data, device="cuda", edge_features=False):
    total = 0
    correct = 0
    model.eval()
    for sample in tqdm(data, desc="Testing"):
        sample = sample.to(device)
        if edge_features:
            if sample.edge_index.shape[1] != sample.edge_features.shape[0]:
                continue
            data = prepare_network_input_data(sample.x, sample.edge_index, sample.edge_features, device=device)
        else:
            data = prepare_network_input_data(sample.x, sample.edge_index, device=device)
        z, emb = model(*data, device) 
        prediction = (z > 0.5).type(torch.int)
        total += len(prediction) 
        correct += sum(prediction == sample.edge_label.type(torch.int))
    return (correct / total)
