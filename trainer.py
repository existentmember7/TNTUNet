import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from datasets.datasets import CustomDataset
from option.option import Option

def Trainer(opt, model):
    training_data = DataLoader(CustomDataset(opt), batch_size=opt.batch_size, shuffle=True)
    model.train()

    writer = SummaryWriter(opt.model_path + '/log')
    logging.basicConfig(filename=opt.model_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(opt))

    ce_loss = BCEWithLogitsLoss().cuda()
    dice_loss = DiceLoss(opt.num_classes).cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iterations = opt.max_iterations * len(training_data)
    iter_num = 0
    iterator = tqdm(range(opt.max_epochs), ncols=70)
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(training_data), max_iterations))

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

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # if iter_num % 20 == 0:
            #     image = i_batch[1, 0:1,:,:]
            #     writer.add_image('train/Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = i_label[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/GroundTruth', labs, iter_num)

        if (epoch_num + 1)%opt.save_interval == 0:
            save_model_path = os.path.join(opt.model_path + 'epoch_'+str(epoch_num)+'.pth')
            torch.save(model.stat_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
    
        if epoch_num >= opt.max_epochs - 1:
            save_model_path = os.path.join(opt.model_path + 'epoch_'+str(epoch_num)+'.pth')
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            iterator.close()
            break
    writer.close()
    return "Finish Training!"