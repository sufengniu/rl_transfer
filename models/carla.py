from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import h5py

from .vgg import VGG
from .resnet import ResNet18
from .cifar_utils import progress_bar
from torch.autograd import Variable

from .dictionary import Dictionary

use_cuda = torch.cuda.is_available()


def load_data(params):
    if params.src_env == '89_nn':
        town1_data = h5py.File('./checkpoint/Town01_89_NN_CSL_img_features.h5', 'r+')
    elif params.src_env == '56_nn':
        town1_data = h5py.File('./checkpoint/Town01_56_NN_CSL_img_features.h5', 'r+')
    elif params.src_env == '12_nn':
        town1_data = h5py.File('./checkpoint/Town01_12_10w_NN_CSL_img_features.h5', 'r+')
    elif params.src_env == '89_ln':
        town1_data = h5py.File('./checkpoint/Town01_89_CSL_img_features_12w.h5', 'r+')
    elif params.src_env == '56_ln':
        town1_data = h5py.File('./checkpoint/Town01_56_v2_CSL_img_features.h5', 'r+')
    elif params.src_env == '12_ln':
        town1_data = h5py.File('./checkpoint/Town01_12_CSL_img_features_10w.h5', 'r+')    
    elif params.src_env == '89_hn':
        town1_data = h5py.File('./checkpoint/Town01_89_CSH_12w_img_features.h5', 'r+')
    elif params.src_env == '56_hn':
        town1_data = h5py.File('./checkpoint/Town01_56_v2_CSH_img_features.h5', 'r+')
    elif params.src_env == '89_ln_t2':
        town1_data = h5py.File('./checkpoint/Town02_89_CSL_img_features.h5', 'r+')


    town1_np = town1_data['feature'].value
    town1_img_np = (np.transpose(town1_data['rgb'].value, [0,3,1,2]) / 255.0).astype(np.float32)

    if params.tgt_env == '89_nn':
        town2_data = h5py.File('./checkpoint/Town01_89_NN_CSL_img_features.h5', 'r+')
    elif params.tgt_env == '56_nn':
        town2_data = h5py.File('./checkpoint/Town01_56_NN_CSL_img_features.h5', 'r+')
    elif params.tgt_env == '12_nn':
        town2_data = h5py.File('./checkpoint/Town01_12_10w_NN_CSL_img_features.h5', 'r+')
    elif params.tgt_env == '89_ln':
        town2_data = h5py.File('./checkpoint/Town01_89_CSL_img_features_12w.h5', 'r+')
    elif params.tgt_env == '56_ln':
        town2_data = h5py.File('./checkpoint/Town01_56_v2_CSL_img_features.h5', 'r+')
    elif params.tgt_env == '12_ln':
        town2_data = h5py.File('./checkpoint/Town01_12_CSL_img_features_10w.h5', 'r+')
    elif params.tgt_env == '89_hn':
        town2_data = h5py.File('./checkpoint/Town01_89_CSH_12w_img_features.h5', 'r+')
    elif params.tgt_env == '56_hn':
        town2_data = h5py.File('./checkpoint/Town01_56_v2_CSH_img_features.h5', 'r+')
    elif params.tgt_env == '89_ln_t2':
        town2_data = h5py.File('./checkpoint/Town02_89_CSL_img_features.h5', 'r+')
        

    town2_np = town2_data['feature'].value
    town2_img_np = (np.transpose(town2_data['rgb'].value, [0,3,1,2]) / 255.0).astype(np.float32)

    return town1_np, town1_img_np, town2_np, town2_img_np


def load_img_embeddings(params, all_np, all_img_np, source, full_vocab=False):

    print('==> Resuming from checkpoint..')

    training_set = torch.from_numpy(all_np[:40000])
    training_img = torch.from_numpy(all_img_np[:40000])

    if use_cuda:
        training_set = training_set.cuda()

    word2id = {}
    for i in range(training_set.shape[0]):
        word2id[i] = i

    lang = params.src_lang if source else params.tgt_lang

    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)

    img = training_img
    embeddings = training_set
    return embeddings, img, dico, None

def get_sample_emb(all_np, all_img_np):

    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'

    testing_set = torch.from_numpy(all_np[40000:])
    testing_img = torch.from_numpy(all_img_np[40000:])
    
    if use_cuda:
        testing_set = testing_set.cuda()

    word2id = {}
    for i in range(testing_set.shape[0]):
        word2id[i] = i

    embeddings = testing_set
    return embeddings, None