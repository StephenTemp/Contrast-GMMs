'''
VAE-MNIST.py : Leveraging a Variational Autoencoder (VAE) to reduce dimensionality
on a simply image dataset (MNIST) using NovelNet
'''

# IMPORTS 
# ---------------------------
from model.NovelNetwork import NovelNetwork
from model.layers.VAE import Encoder, Decoder
import data.utils

import torch
from collections import OrderedDict
# ---------------------------

# CONSTANTS
# ---------------------------
BATCH_SIZE = 64
KKC = [0, 1, 2, 3, 4, 5, 6, 7, 8]              # Classes seen during training and test time (KKC)
UUC = [9]                       # Classes not seen during training time (UUC)
ALL_CLASSES = KKC + UUC         # All classes
PERC_VAL = 0.20                 # Percent of data for validation 
# ---------------------------

def VAEwithMNIST(LATENT_DIMS=3):
    print("Collecting MNIST data . . .\n")
    MNIST = data.utils.get_MNIST(KKC=KKC, ALL=ALL_CLASSES, BATCH_SIZE=BATCH_SIZE)
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
    print_every = 100
    print('using device:', device)

    # Define the layers
    input_dims = 784 # 1 channels x 28 by 28 images
    layers = OrderedDict([
        ("Encoder", Encoder(LATENT_DIMS)),
        ("Decoder", Decoder(LATENT_DIMS))
    ])

    # hyperparameters for our model
    args = {
    'print_every' : 100,
    'feat_layer'  : 'Encoder',
    'feat_sample' : 200,
    'dist_metric' : 'mahalanobis',
    'min_g' : 2,
    'max_g' : 20,
    'epoch' : 15,
    'lr' : 5e-4
    }

    # Run the model
    # ------------------------------------------------
    new_model = NovelNetwork(layers, known_labels=KKC, criterion=torch.nn.CrossEntropyLoss)
    new_model.train(MNIST['TRAIN'], MNIST['VAL'], args, print_info=True)
    # ------------------------------------------------

    new_model.test_analysis(MNIST['TEST'], print_info=True)



if __name__ == "__main__":
    VAEwithMNIST()