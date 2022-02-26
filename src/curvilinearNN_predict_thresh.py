###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

# Python
import numpy as np
import configparser
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from torch.utils import data
import torch.nn.functional as F
# from unet import UNet
from iternet import Iternet
# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import sys

sys.path.insert(0, './lib/')
# help_functions.py
from help_functions import *
# extract_patches.py
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import paint_border
from extract_patches import kill_border
from extract_patches import pred_only_FOV
from extract_patches import get_data_testing
from extract_patches import get_data_testing_overlap
from extract_patches import pred_only_FOV_list
from extract_patches import get_singledata_testing_overlap,get_singledata_testing_overlap_1
# pre_processing.py
from pre_processing import my_PreProc, my_PreProc_three
import pickle
import os
torch.cuda.set_device(0)
# ========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')

# ===========================================
path_data = config.get('data paths', 'path_local')

# original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]

# the border masks provided by the DRIVE
DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')

# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))

# the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)

# model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './' + name_experiment + '/'

# N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))

# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

# ====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')

#ground truth
gtruth= path_data + config.get('data paths', 'test_groundTruth')
test_img_truth= load_hdf5(gtruth)

# SAVE dir
Test_Dataset = str(config.get('testing settings', 'Test_dataset'))
Save_dir = path_experiment + Test_Dataset

if Test_Dataset!='Crack' and Test_Dataset.find('Crack')==-1:
    test_border_masks = load_hdf5(DRIVE_test_border_masks)
else:
    test_border_masks = None

if not os.path.exists(Save_dir):
    os.mkdir(Save_dir)
# ============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test = None
patches_masks_test = None

pred_img_list = []
gt_list = []
mask_list = []
for i in range(0,test_imgs_orig.shape[0]):
    single_image = test_imgs_orig[i]
    single_gt = test_img_truth[i]
    patches_imgs_test, new_height, new_width, masks_test = get_singledata_testing_overlap(single_image, single_gt, patch_height=patch_height, patch_width=patch_width,
                             stride_height=stride_height, stride_width=stride_width)
    # ================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')
    model = Iternet(n_channels=4, n_classes=2).cuda()
    model.load_state_dict(torch.load(name_experiment + '/model/model_best.pth'))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Calculate the predictions
    patches_imgs_test = torch.tensor(patches_imgs_test, dtype=torch.float32)
    print(patches_imgs_test.size())
    test_dataset = data.TensorDataset(patches_imgs_test)
    test_loader = data.DataLoader(test_dataset, batch_size=128, pin_memory=True)
    first = True
    for inputs in tqdm(test_loader):
        inputs = inputs[0]
        inputs = inputs.cuda()
        net_out = model(inputs)  # bs, 2, H, W
        net_out = F.softmax(net_out, dim=1)
        bs, n_class, h, w = net_out.size()
        net_out = net_out.view(bs, n_class, h * w)
        net_out = net_out.permute(0, 2, 1)  # bs, 2304, 2
        net_out = net_out.cpu().numpy()
        if first:
            predictions = net_out
            first = False
        else:
            predictions = np.concatenate((predictions, net_out), axis=0)

    print("predicted images size :")
    print(predictions.shape)
    # ===== Convert the prediction arrays in corresponding images
    pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")

    # ========== Elaborate and visualize the predicted images ====================
    pred_imgs = None
    orig_imgs = None
    gtruth_masks = None
    if average_mode == True:
        pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
        orig_imgs = my_PreProc_three(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals
        gtruth_masks = masks_test  # ground truth masks
    else:
        pred_imgs = recompone(pred_patches, 13, 12)  # predictions
        orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
        gtruth_masks = recompone(patches_masks_test, 13, 12)  # masks
    orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
    pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
    gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]
    if gtruth.find("CHASEDB1") or gtruth.find("STARE"):
        gtruth_masks[gtruth_masks >= 0.5] = 1
        gtruth_masks[gtruth_masks < 0.5] = 0
    with open(f"{Save_dir}/{i + 1:02}_pretrain.pickle",'wb') as handle:
        pickle.dump(pred_imgs[0,0,:,:], handle, protocol=pickle.HIGHEST_PROTOCOL)
    pred_img_list.append(pred_imgs)
    gt_list.append(gtruth_masks)
    if Test_Dataset != 'Crack' and Test_Dataset.find('Crack')==-1:
        mask_list.append(np.expand_dims(test_border_masks[i],axis=0))
    else:
        mask_list.append(test_border_masks)



# ====== Evaluate the results
print("\n\n========  Evaluate the results =======================")
# predictions only inside the FOV
y_scores, y_true = pred_only_FOV_list(pred_img_list, gt_list, mask_list, Test_Dataset)  # returns data only inside the FOV
print("Calculating results only inside the FOV:")
print("y scores pixels: " + str(y_scores.shape[0]) +
      " (radius 270: 270*270*3.14==228906), including background around retina: " +
      str(pred_imgs.shape[0] * pred_imgs.shape[2] * pred_imgs.shape[3]) + " (584*565==329960)")
print("y true pixels: " + str(y_true.shape[0]) +
      " (radius 270: 270*270*3.14==228906), including background around retina: " +
      str(gtruth_masks.shape[2] * gtruth_masks.shape[3] * gtruth_masks.shape[0]) + " (584*565==329960)")

# Area under the ROC curve
print("y_true",np.unique(y_true))
fpr, tpr, thresholds = roc_curve((y_true), y_scores)
AUC_ROC = roc_auc_score(y_true, y_scores)
print("\nArea under the ROC curve: " + str(AUC_ROC))

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
AUC_prec_rec = np.trapz(precision, recall)
print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))

# Confusion matrix
F1_max = 0
for thresh_value in range(0, 100):
    threshold_confusion = thresh_value / 100
    print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_score(y_true, y_pred)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))

    # Save the results
    # Need change the Save Name
    file_perf = open(path_experiment +Test_Dataset+'/'+str(threshold_confusion)+ 'performances_DRIVE_pretrain_test.txt', 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                    + "\nJaccard similarity score: " + str(jaccard_index)
                    + "\nF1 score (F-measure): " + str(F1_score)
                    + "\n\nConfusion matrix:"
                    + str(confusion)
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    )
    file_perf.close()

    if F1_score>=F1_max:
        file_best = open(path_experiment + Test_Dataset + '/' + str(
            threshold_confusion) + 'performances_DRIVE_test_pretrainbest.txt', 'w')
        file_best.write("Area under the ROC curve: " + str(AUC_ROC)
                        + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                        + "\nJaccard similarity score: " + str(jaccard_index)
                        + "\nF1 score (F-measure): " + str(F1_score)
                        + "\n\nConfusion matrix:"
                        + str(confusion)
                        + "\nACCURACY: " + str(accuracy)
                        + "\nSENSITIVITY: " + str(sensitivity)
                        + "\nSPECIFICITY: " + str(specificity)
                        + "\nPRECISION: " + str(precision)
                        )
        F1_max = F1_score
        file_best.close()