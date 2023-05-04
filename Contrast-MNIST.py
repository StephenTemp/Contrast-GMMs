'''
Contrast-MNIST.py : Leveraging a contrastive network to reduce dimensionality
on a simply image dataset (MNIST) using NovelNet
'''

# IMPORTS 
# ---------------------------
from model.NovelNetwork import NovelNetwork
from model.layers.MVNet import MV_CNN
from model.layers.SupConLoss import SupConLoss
import data.utils

import torch
from collections import OrderedDict
# ---------------------------

# CONSTANTS
# ---------------------------
BATCH_SIZE = 64
KKC = ["airplane", "toilet", "guitar", "bed"]  # Classes seen during training and test time (KKC)
UUC = ["car"]                           # Classes not seen during training time (UUC)
ALL_CLASSES = KKC + UUC                 # All classes
PERC_VAL = 0.20                     # Percent of data for validation 
# ---------------------------

def CNTwithMNIST(LATENT_DIMS=3):
    print("Collecting MNIST data . . .\n")
    MNIST = data.utils.get_ModelNet(KKC=KKC, ALL=ALL_CLASSES, BATCH_SIZE=BATCH_SIZE)
    print(". . . done!")

    # Show examples from MNIST
    data.utils.showImg2d(MNIST['TRAIN'])
    # Set the device (CUDA compatibility needs to be added to NovelNetwork.py)
    USE_GPU = True
    dtype = torch.float32 # we will be using float throughout this tutorial
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Constant to control how frequently we print train loss
    print('using device:', device)

    # Define the layers
    layers = MV_CNN(LATENT_DIMS)
    # hyperparameters for our model
    args = {
        'print_every' : 100,
        'feat_sample' : 200,
        'feat_layer' : 'fc2',
        'dist_metric' : 'mahalanobis',
        'epoch' : 5,
        'lr' : 1e-4
    }
    
    lrs = [1e-4, 5e-5, 1e-5, 5e-6]
    acc_dict = {}
    for lr in lrs:
        args['lr'] = lr
        # Run the model
        # ------------------------------------------------
        new_model = NovelNetwork(layers, known_labels=KKC, criterion=SupConLoss) # UPDATE THIS
        acc = new_model.train(MNIST['TRAIN'], MNIST['VAL'], args, print_info=True)
        # ------------------------------------------------
        acc_dict[lr] = acc
        #new_model.test_analysis(MNIST['TEST'], print_info=True)

    print(acc_dict)

if __name__ == "__main__":
    CNTwithMNIST()