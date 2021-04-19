import argparse
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from datasets.datasets import CustomDataset
from option.option import Option

def Trainer(opt, model):
    training_data = DataLoader(CustomDataset(opt), batch_size=opt.batch_size, shuffle=True)
    model.train()
    ce_loss = BCEWithLogitsLoss().cuda()
    dice_loss = DiceLoss(opt.num_classes).cuda()
    optimizer = optim.SGD(model.parameters(), lr=opt.base_lr, momentum=0.9, weight_decay=0.0001)
    max_iterations = opt.max_iterations * len(training_data)
    iter_num = 0
    iterator = tqdm(range(opt.max_epochs), ncols=70)
    for epoch_num in iterator:
        for i, (i_batch, i_label) in enumerate(training_data):
            i_batch, i_label = i_batch.cuda(), i_label.type(torch.FloatTensor).cuda()
            outputs = model(i_batch)
            loss_ce = ce_loss(outputs, i_label)
            loss_dice = dice_loss(outputs, i_label[:], softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = opt.base_lr *(1.0 - iter_num/max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num += 1