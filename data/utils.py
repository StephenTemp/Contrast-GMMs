''' 
utils.py : data utility functions to reduce clutter
'''

# IMPORTS
# ------------------------------------------
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------

# func: train_ind() => []
# return indices matching non-novel classes
def train_ind(dataset, CLASSES):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in CLASSES:
            indices.append(i)

    return indices


# func: test_ind() => []
# return indices matching non-novel classes 
#                         + novel classes
def test_ind(dataset, CLASSES, PERC_VAL=0.20, val=False):
    indices =  []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in CLASSES:
            indices.append(i)
    
    # if this is the validation set, return PERC_VAL% of the data      
    if val == True: return indices[:int(PERC_VAL * len(indices))]
    else: return indices[int(PERC_VAL * len(indices)):]


def get_MNIST(KKC, ALL, BATCH_SIZE=64, VAL_PERC=0.20):

    trainset = torchvision.datasets.MNIST(root='./data', download=True, 
                                     transform=torchvision.transforms.ToTensor())

    train_inds = train_ind(trainset, CLASSES=KKC)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler = SubsetRandomSampler(train_inds))


    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, 
                                        transform=torchvision.transforms.ToTensor())
    val_inds = test_ind(testset, CLASSES=ALL, val=True, PERC_VAL=VAL_PERC)
    val_loader = DataLoader(testset, batch_size=BATCH_SIZE, sampler = SubsetRandomSampler(val_inds))

    test_inds = test_ind(testset, CLASSES=ALL, PERC_VAL=VAL_PERC)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE,
                                            sampler = SubsetRandomSampler(test_inds))

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    return { "TRAIN" : train_loader, "VAL" : val_loader, "TEST" : test_loader, "CLASSES" : classes }


def showImg2d(dataloader):
    # FUNC: imshow( [] ) => None
    # SUMMARY: visualize the sampels
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    print(images[0].shape)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    labels = np.array(labels)
    df = pd.DataFrame(labels.reshape( (int(np.sqrt(len(images))), int(np.sqrt(len(images))) )))
    print(df)