# System modules
import sys
from glob import glob
from copy import copy
from time import time

# Standard python modules
import numpy as np
import matplotlib.pyplot as plt
from tqdm  import tqdm
from scipy import ndimage
from numpy import fft
from numpy.random import RandomState

# EHT modules
import ehtim   as eh
import ehtplot as ep

# Feature engines
#from keras.models import load_model, Model
#from keras.applications.vgg16 import VGG16, preprocess_input
#from keras.preprocessing import image

from sklearn.decomposition import PCA #, KernelPCA, SparsePCA, TruncatedSVD, IncrementalPCA, NMF, FastICA, MiniBatchSparsePCA, FactorAnalysis, MiniBatchDictionaryLearning
from sklearn.manifold      import TSNE

# Clustering algorithms
from sklearn.cluster import KMeans#, MiniBatchKMeans
#from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

#from sklearn.metrics import silhouette_samples, silhouette_score

#==============================================================================
# Reprocessing: img are 2d numpy array

def cutoff(img, ratio=0.99):
    """
    Solve for the flux cutoff such that `ratio` amount of flux is captured
    by the brightnest pixels.
    """
    ls = np.sort(img, axis=None)
    cs = np.cumsum(ls)
    return ls[np.argmax(cs >= (1-ratio) * np.sum(ls))]

def relevant(img, norm=None, snr=None, cfr=None):
    if snr is None and cfr is None:
        cfr = 0.99 # set default cutoff ratio if snr and cfr are not set

    if norm:
        img /= np.max(img)

    if cfr is not None:
        c = cutoff(img, ratio=cfr)
    elif snr is not None:
        c = snr * np.max(img)
    else:
        raise ValueError('`snr` and `cfr` cannot be set simultaneously.')

    return np.where(img >= c, img, 0)

def corr(img1, img2):
    vis1 = fft.fftn(img1)
    vis2 = fft.fftn(img2)
    return np.max(np.real(fft.ifftn(vis1 * np.conj(vis2)))) / np.size(img1)

def nxcorr(img1, img2):
    return (corr(img1, img2) - np.mean(img1) * np.mean(img2)) / (np.std(img1) * np.std(img2))

def nxcinf(img1, img2):
    return corr(img1, img2) / np.sqrt(np.mean(img1*img1) * np.mean(img2*img2))

def align(ref, img):
    vis1 = fft.fftn(ref)
    vis2 = fft.fftn(img)
    c    = np.real(fft.ifftn(vis1 * np.conj(vis2)))
    i, j = np.unravel_index(np.argmax(c), c.shape)
    return np.roll(np.roll(img, i, axis=0), j, axis=1)

#==============================================================================
def to_img(im, xdim=80, psize=9.696274165524704e-12, ref=None):
    imvec = np.array(im.imvec).reshape(-1)
    #fovfactor  = im.xdim * im.psize / eh.RADPERUAS
    #unitfactor = 1.0e3 * im.xdim**2 / fovfactor**2 #flux unit is mJy/uas^2
    #img        = ndimage.zoom(imvec.reshape(im.ydim, im.xdim), im.psize / psize, order=1) * unitfactor

    img = ndimage.zoom(imvec.reshape(im.ydim, im.xdim), im.psize / psize, order=1) * (psize / im.psize)**2
    ret = np.zeros((xdim, xdim))
    xsh = img.shape[0]
    ysh = img.shape[1]
    ret[(xdim-xsh)//2:(xdim+xsh)//2,
        (xdim-ysh)//2:(xdim+ysh)//2] = img

    if ref is None:
        return ret
    else:
        return align(ref, ret)

def load(dir, n=256, individual=False, ref=None):
    allfiles = glob(dir+'/*.fits')
    np.random.shuffle(allfiles)
    if isinstance(n, int):
        N = len(allfiles)
        if n < N//2:
            # easiler to find selection index
            js = np.unique(np.random.randint(0, N, size=int(n * 1.25) + 64))[:n]
            files = [allfiles[j] for j in js]
        else:
            # easiler to find rejection index
            js = np.unique(np.random.randint(0, N, size=int((N-n) * 1.25) + 64))[:N-n]
            files = [f for j, f in enumerate(allfiles) if j not in js]
    else:
        files = allfiles

    saved_stdout = sys.stdout
    sys.stdout   = None
    imgs = {f:to_img(eh.image.load_fits(f), ref=ref) for f in tqdm(files)}
    sys.stdout   = saved_stdout

    if individual:
        return {f:np.where(img >= cutoff(img, 0.99), img, 0) / cutoff(img, 0.01) for f, img in imgs.items()}
    else:
        upper = np.median([cutoff(img, 0.01) for img in imgs.values()])
        lower = np.median([cutoff(img, 0.99) for img in imgs.values()])
        return {f:np.where(img >= lower, img, 0) / upper for f, img in imgs.items()}

#==============================================================================
def plot_grid(imgs, ncol=16, sz=None, vmin=0, vmax=1, cmap=None):
    if not sz:
        sz = list(imgs)[0].shape[0]
    nrow = (len(imgs)-1) // ncol + 1
    grid = np.zeros((nrow*sz, ncol*sz))
    for h, img in enumerate(imgs):
        i = h %  ncol
        j = h // ncol
        grid[j*sz:(j+1)*sz, i*sz:(i+1)*sz] = img

    if cmap is None:
        cmap = copy(plt.get_cmap('afmhot_10us'))
        cmap.set_over('b')

    plt.figure(figsize=(ncol,nrow))
    plt.imshow(grid, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis('off')

#==============================================================================
# class CNN:
#     """Keras Model of the VGG16 network, with the output layer set to `layer`.
#     The default layer is the second-to-last fully connected layer 'fc2' of
#     shape (4096,).
#     Parameters
#     ----------
#     layer : str
#         which layer to extract (must be of shape (None, X)), e.g. 'fc2', 'fc1'
#         or 'flatten'
#     """
#     # base_model.summary():
#     #     ....
#     #     block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
#     #     _________________________________________________________________
#     #     block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
#     #     _________________________________________________________________
#     #     block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
#     #     _________________________________________________________________
#     #     block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
#     #     _________________________________________________________________
#     #     block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
#     #     _________________________________________________________________
#     #     block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
#     #     _________________________________________________________________
#     #     flatten (Flatten)            (None, 25088)             0
#     #     _________________________________________________________________
#     #     fc1 (Dense)                  (None, 4096)              102764544
#     #     _________________________________________________________________
#     #     fc2 (Dense)                  (None, 4096)              16781312
#     #     _________________________________________________________________
#     #     predictions (Dense)          (None, 1000)              4097000

#     def __init__(self):
#         base_model = VGG16(weights='imagenet', include_top=True) # Using pre-training on ImageNet
#         self.model = Model(inputs=base_model.input,
#                            outputs=base_model.get_layer('fc1').output)

#     def feature_vector(img, model):
#         top          = 0.5
#         img          = 255 * np.minimum(img/top, 1.0)
#         img_rs       = ndimage.zoom(img, 2.8, order=1)
#         img_rs_4d    = img_rs[np.newaxis,:,:,np.newaxis].repeat(3, axis=3)
#         img_rs_4d_pp = preprocess_input(img_rs_4d)
#         return model.predict(img_rs_4d_pp)[0,:]

#     def feature_vectors(imgs_dict, model):
#         return {f:feature_vector(img, model) for f, img in tqdm(imgs_dict.items())}

# ref_path = 'ehtim/casa_3599_SGRA_lo_netcal_LMTcal_scan_00sys/casa_3599_SGRA_lo_netcal_LMTcal_scan_00sys_11651_dsct_static.fits'

# rng = RandomState(0)

# estimators = [
#     ('CNN: VGG16 with imagenet',
#      CNN(),
#      False),

#     ('Eigenimages - PCA',
#      PCA(whiten=True),
#      True),

#     ('Eigenimages - PCA using randomized SVD',
#      PCA(whiten=True, svd_solver='randomized'),
#      True),

#     ('Independent components - FastICA',
#      FastICA(whiten=True),
#      True),

#     ('Non-negative components - NMF',
#      NMF(init='nndsvda', tol=5e-3),
#      False),

#     ('Sparse comp. - MiniBatchSparsePCA',
#      MiniBatchSparsePCA(alpha=0.8, n_iter=100, batch_size=3, random_state=rng),
#      True),

#     ('Cluster centers - MiniBatchKMeans',
#      MiniBatchKMeans(n_clusters=32, tol=1e-3, batch_size=20, max_iter=50, random_state=rng),
#      True),

#     ('Factor Analysis components - FA',
#      FactorAnalysis(max_iter=20),
#      True),

#     ('Dictionary learning',
#      MiniBatchDictionaryLearning(alpha=0.1, n_iter=50, batch_size=3,                     random_state=rng),
#      True),

#     ('Dictionary learning - positive dictionary',
#      MiniBatchDictionaryLearning(alpha=0.1, n_iter=50, batch_size=3,                     random_state=rng, positive_dict=True),
#      True),

#     ('Dictionary learning - positive code',
#      MiniBatchDictionaryLearning(alpha=0.1, n_iter=50, batch_size=3, fit_algorithm='cd', random_state=rng,                     positive_code=True),
#      True),

#     ('Dictionary learning - positive dictionary & code',
#      MiniBatchDictionaryLearning(alpha=0.1, n_iter=50, batch_size=3, fit_algorithm='cd', random_state=rng, positive_dict=True, positive_code=True),
#      True),
# ]
