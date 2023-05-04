'''
MV-ModelNet.py: Running a contrastive network on ModelNet40 dataset using a 
                Multi-View CNN approach
'''

# IMPORTS 
# ---------------------------
from model.NovelNetwork import NovelNetwork
from model.layers.SupConLoss import SupConLoss
from model.layers.MVNet import MV_CNN
import data.utils

import torch
from collections import OrderedDict
# ---------------------------

# CONSTANTS
# ---------------------------
BATCH_SIZE = 64
KKC = ["airplane", "toilet", "guitar"]  # Classes seen during training and test time (KKC)
UUC = ["car"]                           # Classes not seen during training time (UUC)
ALL_CLASSES = KKC + UUC                 # All classes
PERC_VAL = 0.20                         # Percent of data for validation 
# ---------------------------

def MV_ModelNet(LATENT_DIMS=3):
    print("Collecting ModelNet data . . .\n")
    MODELNET = data.utils.get_ModelNet(KKC=KKC, ALL=ALL_CLASSES, BATCH_SIZE=BATCH_SIZE)
    print(". . . done!")

    # Show examples from MNIST
    data.utils.showImg2d(MODELNET['TRAIN'])
    # Set the device (CUDA compatibility needs to be added to NovelNetwork.py)
    USE_GPU = True
    dtype = torch.float32 # we will be using float throughout this tutorial
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')


    # Constant to control how frequently we print train loss
    print_every = 100
    print('using device:', device)

    # Define the layers
    layers = MV_CNN(LATENT_DIMS)

    # hyperparameters for our model
    args = {
    'print_every' : 100,
    'feat_layer'  : 'fc',
    'feat_sample' : 50,
    'dist_metric' : 'mahalanobis',
    'epoch' : 1,
    'lr' : 5e-4
    }

    # Run the model
    # ------------------------------------------------
    new_model = NovelNetwork(layers, known_labels=[i + 1 for i in range(0, len(KKC))], criterion=SupConLoss)
    new_model.train(MODELNET['TRAIN'], MODELNET['VAL'], args, print_info=True)
    # ------------------------------------------------

    new_model.test_analysis(MODELNET['TEST'], print_info=True)



if __name__ == "__main__":
    MV_ModelNet()