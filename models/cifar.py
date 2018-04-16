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

from .vgg import VGG
from .resnet import ResNet18
from .cifar_utils import progress_bar
from torch.autograd import Variable

from .dictionary import Dictionary

use_cuda = torch.cuda.is_available()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_img_embeddings(params, source, full_vocab=False):

    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if source == True:
        checkpoint = torch.load('./checkpoint/vgg.t7')
    else:
        checkpoint = torch.load('./checkpoint/resnet.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    # Data
    print('==> Preparing data..')
    trainset = torchvision.datasets.CIFAR10(root='/scratch2/sniu/cifar10_data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    test_batch = 100
    testset = torchvision.datasets.CIFAR10(root='/scratch2/sniu/cifar10_data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=2)

    print('==> Building model..')
    if source == True:
        net = VGG('VGG19')
    else:
        net = ResNet18()

    if use_cuda:
        net.cuda()

    word2id = {}
    dictionary = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs, features = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        for i in range(test_batch):
            word2id[i+batch_idx*test_batch] = i+batch_idx*test_batch

        dictionary.append(features.data)

    lang = params.src_lang if source else params.tgt_lang

    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)

    dictionary = torch.stack(dictionary, dim=0)
    embeddings = dictionary.view(-1, dictionary.shape[-1])
    return embeddings, dico

# def get_sample_emb():

#     num_sample = 16
#     trainset = torchvision.datasets.CIFAR10(root='/scratch2/sniu/cifar10_data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         if use_cuda:
#             inputs, targets = inputs.cuda(), targets.cuda()

#         inputs, targets = Variable(inputs, volatile=True), Variable(targets)
#         outputs, features = net(inputs)

#         _, predicted = torch.max(outputs.data, 1)
#         for i in range(test_batch):
#             word2id[i+batch_idx*test_batch] = i+batch_idx*test_batch

#         dictionary.append(features.data)
