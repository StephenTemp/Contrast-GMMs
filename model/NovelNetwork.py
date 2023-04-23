# IMPORTS
# ------------------------------------------
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis

import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------
IS_UCC = 1
IS_KKC = 0

class NovelNetwork(torch.nn.Module):
    # instance variables
    # ------------------
    model = None            # Architecture (arbitrary)
    gmm = None              # GMM model for neural outputs 
    device = None           # for GPU

    criterion = None        # Loss Criterion (network-dependent)
    feat_layer = None       # Feature map to extract from model
    known_labels = None     # Specify which labels are KKC
    threshold = None        # Distance threshold to classify as 'novel'
    dist_metric = None      # Distance metric (default: mahalanobis)
    confidence_delta = None

    # ----------------------------------------------------
    # Nearest-Class Mean    # (this is another baseline)
    NearCM = None
    NearCM_delta = None
    
    def __init__(self, model, known_labels, criterion, use_gpu=True):
        super().__init__()
        self.criterion = criterion
        self.known_labels = torch.tensor(known_labels)
        
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.model = model
    
    # FUNCTION: extract_feats( [], str ) => []
    # SUMMARY: provided a target layer and input, return feats from
    #          the target layer
    def extract_feats(self, input, target_layer):
        model = self._modules['model']
        cur_feats = input
        for cur_layer in model._modules:
            cur_feats = model._modules[cur_layer](cur_feats)
            if cur_layer == target_layer: return cur_feats
        
        raise ValueError("[Feature Extraction] Request layer not found!")

    def to_novel(self, input, label_kkc=False):
        input = input.to(dtype=int)
        bools = torch.tensor([x not in self.known_labels for x in input])

        if not label_kkc: input[bools] = IS_UCC
        else: return bools.to(dtype=int)

    def raw_predict(self, input):
        model = self._modules['model']
        return model(input).argmax(1)

    def predict(self, input):
        if self.threshold == None: 
            raise ValueError("Model is not yet trained!")

        raw_preds = np.array(self.raw_predict(input))
        feats = np.array(self.extract_feats(input, self.feat_layer).detach())
        gmm_preds = self.gmm.predict(feats)
        gmm_means = self.gmm.means_[gmm_preds]
        gmm_cov = self.gmm.covariances_

        std_dist = np.zeros(shape=feats.shape[0])
        for i, sample_cov in enumerate(gmm_cov):
            sample_inv = np.linalg.inv(sample_cov)
            sample_dist = mahalanobis(feats[i], gmm_means[i], sample_inv)
            std_dist[i] = sample_dist

        raw_preds[std_dist > self.threshold] = -1
        return raw_preds
    
    # FUNCTION: augment( x ) => x'
    # SUMMARY: provided some samples, augment them with respect to 
    #          color, rotation, etc.
    def augment(self, x):
        transforms = torch.nn.Sequential(
            torchvision.transforms.RandomCrop(x.shape[2]),
            torchvision.transforms.ColorJitter(),
        )

        trans = torch.jit.script(transforms)
        aug_x = trans(x)
        return aug_x

    # FUNCTION: train( y, {} ) => void
    # SUMMARY: train the network, then fit a GMM to a sample of the 
    #          training data features
    def train(self, train_data, val_data, args, print_info=False):
        self.feat_layer = args['feat_layer']
        criterion = self.criterion()
        if 'dist_metric' in args:
            self.dist_metric = args['dist_metric']
        else: self.dist_metric = 'mahalanobis'

        print_every = args['print_every']
        feat_samp = args['feat_sample']
        min_clusters = args['min_g']
        max_clusters = args['max_g']
        epochs = args['epoch']
        lr = args['lr']

        # train neural network
        # ---------------------
        model = self._modules['model']
        device = self.device
        print(model._parameters)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = model.to(device=self.device)  # move the model parameters to CPU/GPU

        for _ in range(epochs):
            for t, (x, y) in enumerate(train_data):
                model.train()  # put model to training mode
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                # Data Augmentation
                aug_x = self.augment(x)

                scores = model(x)
                scores_aug = model(aug_x)

                feats = torch.zeros(size=(scores.shape[0], 2, scores.shape[1]))
                feats[:, 0] = scores
                feats[:, 1] = scores_aug

                loss = criterion(feats, labels=y) #criterion(scores, y) #F.cross_entropy(scores, y)

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()

                if t % print_every == 0 and print_info==True:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    #self.check_accuracy(val_data, model)
                    print()
        
        # run coarse search over gaussian mixture model
        # ---------------------------------------------
        X_feats, _, y = self.GMM_batch(train_data, self.feat_layer, feat_samp)

        best_aic = float('inf')
        best_gmm = None
        for n_comp in range(min_clusters, max_clusters):
            cur_gmm = GaussianMixture(n_components=n_comp)
            cur_gmm.fit(X_feats, y)
            cur_aic = cur_gmm.aic(X_feats)
            if cur_aic < best_aic:
                best_aic = cur_aic
                best_gmm = cur_gmm

        self.gmm = best_gmm
        print("Best # components: ", best_gmm.n_components)
        
        # Set the mahalanobis threshold 
        # ------------------------------
        X_test_feats, X_test, y_test = self.GMM_batch(val_data, self.feat_layer, feat_samp)
        best_acc = self.set_threshold(X_test, X_test_feats, y_test)
        return best_acc
    

    def GMM_batch(self, loader, target_layer, num_batches=50):

        X_feats = np.array([])
        X = torch.tensor([])
        y = torch.tensor([], dtype=torch.float32)
        for i in range(0, num_batches):
            cur_batch = iter(loader)
            X_batch, y_batch = next(cur_batch)

            X_batch = X_batch.to(self.device)
            cur_feats = self._modules['model'](X_batch) #self.extract_feats(X_batch, target_layer)

            X = torch.cat((X.cpu(), X_batch.cpu()), axis=0)
            y = torch.cat((y.cpu(), y_batch.cpu()), axis=0)

            if i == 0: X_feats = X_feats.reshape((0, cur_feats.shape[1]))
            X_feats = np.concatenate((X_feats, cur_feats.detach().cpu()), axis=0)

        return X_feats, X, y


    def check_accuracy(self, loader, get_wrong=False):
        model = self._modules['model']
        device = self.device

        if loader.dataset.train: print('Checking accuracy on validation set')
        else: print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        wrong_imgs = []
        wrong_labels = []
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=torch.long)
                scores = model(x)
                _, preds = scores.max(1)

                img_wr = x[preds != y].to('cpu')
                label_wr = preds[preds != y].to('cpu')
                wrong_imgs.extend(img_wr)
                wrong_labels.extend(label_wr)

                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
        if get_wrong == True: 
            return wrong_imgs, np.array(wrong_labels)
    

    def set_threshold(self, X, X_test_feats, y_test, print_info=False):
        dist_metric = self.dist_metric
        y_novel = np.array(self.to_novel(y_test, label_kkc=True))

        gmm_preds = self.gmm.predict(X_test_feats)
        gmm_means = self.gmm.means_[gmm_preds]
        gmm_cov = self.gmm.covariances_

        # Calculate distance between instance and its clsoest center
        # ----------------------------------------------
        std_dist = np.zeros(shape=X_test_feats.shape[0])
        for i, sample in enumerate(X_test_feats):
            cur_sample = sample.reshape(-1, 1)
            cur_mean = gmm_means[i].reshape(gmm_means[i].shape[0], 1)
            
            iv = np.linalg.inv(gmm_cov[gmm_preds[i]])

            if dist_metric == 'euclidean': sample_dist = np.sum(np.abs(cur_sample - cur_mean))
            elif dist_metric == 'mahalanobis': sample_dist = mahalanobis(cur_sample, cur_mean, iv)
            else: raise ValueError('unsupported distance metric')
            std_dist[i] = sample_dist
        # ----------------------------------------------

        # find the best novelty threshold
        min_thresh = min(std_dist)
        max_thresh = max(std_dist)
        threshold = min_thresh
        thresh_delta = abs(max_thresh - min_thresh)/1000
        
        cur_it = 0
        best_acc = - float('inf')
        best_threshold = None
        while threshold < max_thresh:
            cur_preds = np.copy(gmm_preds)
            UUC_inds = std_dist > threshold
            KKC_inds = std_dist <= threshold

            cur_preds[UUC_inds] = IS_UCC
            cur_preds[KKC_inds] = IS_KKC

            cur_acc = np.sum(cur_preds == y_novel) / len(y_novel)
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_threshold = threshold

            if print_info==True:
                for i, label in enumerate(['novel', 'known']):
                    cur = i - 1
                    plt.scatter(X_test_feats[cur_preds == cur, 0], X_test_feats[cur_preds == cur, 1], label=label)
                    plt.scatter(gmm_means[:, 0], gmm_means[:, 1], marker='x', color='red')
                    #plt.title('Delta = {DELTA}, Acc = {ACC}'.format(DELTA=best_threshold, ACC=best_acc))

                plt.legend()
                plt.axis('off')
                plt.savefig('train-plot.jpeg')
                plt.show()

            threshold += thresh_delta
            cur_it = cur_it + 1

        # plot best thresholding
        cur_preds = np.copy(gmm_preds)
        cur_preds[std_dist > best_threshold] = IS_UCC
        cur_preds[std_dist <= best_threshold] = IS_KKC


        if True:#print_info==True:
            for i, label in enumerate(['known', 'novel']):
                cur = i
                plt.subplot(1, 2, 1)
                plt.scatter(X_test_feats[cur_preds == cur, 0], X_test_feats[cur_preds == cur, 1], label=label)     
                plt.scatter(gmm_means[:, 0], gmm_means[:, 1], marker='x', color='red')
            plt.legend()
            plt.axis('off')

            y_test = y_test.numpy()
            for i, label in enumerate(self.known_labels.numpy()):
                cur = i
                plt.subplot(1, 2, 2)
                plt.scatter(X_test_feats[y_test == cur, 0], X_test_feats[y_test == cur, 1], label=label)     
            plt.scatter(X_test_feats[y_novel == 1, 0], X_test_feats[y_novel == 1, 1], label="novel") 
            
            plt.legend()
            plt.axis('off')
            plt.savefig('train-plot.jpeg')
            plt.show()   
        return best_acc

    def test_analysis(self, loader, get_wrong=False, print_info=False):
        model = self._modules['model']
        dist_metric = self.dist_metric
        X_test_feats, X_test, y_test = self.GMM_batch(loader, self.feat_layer, 20)
        y_test = np.array(self.to_novel(y_test))
        y_test = np.array(y_test)

        preds = torch.tensor(self.predict(X_test))
        preds = np.array(self.to_novel(preds))

        gmm_preds = self.gmm.predict(X_test_feats)
        gmm_means = self.gmm.means_[gmm_preds]
        gmm_cov = self.gmm.covariances_[gmm_preds]
        std_dist = np.zeros(shape=X_test_feats.shape[0])
        for i, sample in enumerate(X_test_feats):
            cur_sample = sample.reshape(-1, 1)
            cur_mean = gmm_means[i].reshape(gmm_means[i].shape[0], 1)
            
            iv = np.linalg.inv(gmm_cov[gmm_preds[i]])
            if dist_metric == 'euclidean': sample_dist = np.sum(np.abs(cur_sample - cur_mean))
            elif dist_metric == 'mahalanobis': sample_dist = mahalanobis(cur_sample, cur_mean, iv)
            else: raise ValueError('unsupported distance metric')
            
            std_dist[i] = sample_dist
        
        preds[std_dist > self.threshold] = IS_UCC

        acc = np.sum(preds == y_test) / len(y_test)
        preds[preds > IS_UCC] = 0

        if print_info==True:
            for i, label in enumerate(['novel', 'known']):
                    cur = i - 1
                    plt.figure('Test Analysis')
                    plt.scatter(X_test_feats[preds == cur, 0], X_test_feats[preds == cur, 1], label=label)
                            
                    plt.scatter(self.gmm.means_[:, 0], self.gmm.means_[:, 1], marker='x', color='red')
                    plt.title('Delta = {DELTA}, Acc = {ACC}'.format(DELTA=self.threshold, ACC=acc))
                    plt.legend()
            plt.savefig('test-plot.jpeg')
            plt.show()

            # Ground truth plot
            plt.figure('Ground Truth')
            plt.scatter(X_test_feats[y_test == IS_NOVEL, 0], X_test_feats[y_test == IS_NOVEL, 1], label='novel')
            plt.scatter(X_test_feats[np.where((y_test == 0) | (y_test == 1)), 0], X_test_feats[np.where((y_test == 0) | (y_test == 1)), 1], label='known')

            plt.scatter(self.gmm.means_[:, 0], self.gmm.means_[:, 1], marker='x', color='red')
            plt.title('Delta = {DELTA}, Acc = {ACC}'.format(DELTA=self.threshold, ACC=acc))
            plt.legend()
            plt.savefig('test-truth-plot.jpeg')
            plt.show()

        info = {}

        raw_preds = np.array(self.raw_predict(X_test))
        raw_acc = np.sum(raw_preds == y_test) / len(y_test)
        true_novel = y_test[y_test == -1]

        # compute recall on novel class
        # ------------------------------
        pred_novel = preds[y_test == -1]
        info['novel_recall'] =  np.sum(pred_novel == true_novel) / len(true_novel)
        # ------------------------------

        return acc, raw_acc, info


    def plot_feats(self, data, target_layer):
        X_feats, X, y = self.GMM_batch(data, target_layer, 20)
        y = self.to_novel(y)
        y[y != IS_NOVEL] = 0 

        plt.figure()
        plt.scatter(X_feats[y == IS_NOVEL, 0], X_feats[y == IS_NOVEL, 1], label='novel')
        plt.scatter(X_feats[y == 0, 0], X_feats[y == 0, 1], label='known')

        plt.axis('off')
        plt.savefig('feats-{layer}.jpeg'.format(layer=target_layer))
