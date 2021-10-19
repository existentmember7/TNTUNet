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
from utils import *
from datasets.datasets import CustomDataset
from option.option import Option

def inference(model, validating_data, ignore_background, n_classes):
    # print("class: ", n_classes)
    ds = validating_data.dataset
    model.eval()
    mIoUs = []
    device = torch.device("cuda:0")
    for i, (i_batch, i_label) in tqdm(enumerate(validating_data)):
        i_batch, i_label = i_batch.cuda().to(device), i_label.type(torch.FloatTensor).cuda().to(device)
        outputs = model(i_batch)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

        for i in range(len(outputs)):
            mIoU = IoU(outputs[i], i_label[i], n_classes, ignore_background)
            mIoUs.append(mIoU)
    mean_IoU = np.mean(np.array(mIoUs))
    return mean_IoU

def validate(model, writer, epoch_num, validating_data ,n_class, ignore_background, learning_log, iter, loss):
    mean_IoU = inference(model, validating_data, ignore_background, n_class)
    writer.add_scalar('info/val_IoU', mean_IoU, epoch_num)
    learning_log.write("train,"+str(iter)+","+str(loss)+","+mean_IoU)
    return writer

def Trainer(opt, model):
    training_data = DataLoader(CustomDataset(opt), batch_size=opt.batch_size, shuffle=True)
    if opt.validating_data_path != None:
        validating_data = DataLoader(CustomDataset(opt, val=True), batch_size=1, shuffle=True)
    model.train()

    writer = SummaryWriter(opt.model_path + 'log')
    logging.basicConfig(filename=opt.model_path + "log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger()#.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(opt))

    ce_loss = BCEWithLogitsLoss().cuda()
    dice_loss = DiceLoss(opt.num_classes).cuda()
    focal_loss = FocalLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=opt.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iterations = opt.max_iterations * len(training_data)
    iter_num = 0
    iterator = tqdm(range(opt.max_epochs), ncols=70)

    last_epoch = 0
    if opt.continue_training:
        last_epoch = int(opt.model_weight_path.split('/')[-1].split('.')[0].split('_')[-1])+1
        iter_num = int(len(training_data)*last_epoch)

    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(training_data), max_iterations))
    
    device = torch.device("cuda:0")

    learning_log = open("./model/learning_log.txt","a")
    learning_log.writelines("type,iter,loss,acc")

    for epoch_num in iterator:
        print("epoch " + str(epoch_num))
        for i, (i_batch, i_label) in tqdm(enumerate(training_data)):
            i_batch, i_label = i_batch.cuda().to(device), i_label.type(torch.FloatTensor).cuda().to(device)
            outputs = model(i_batch)

            # print("outputs size: ", outputs.size())
            # print("label size: ", i_label.size())
            # print("label[:] size: ", i_label[:].size())
            # exit(-1)

            # loss_ce = ce_loss(outputs, i_label)
            loss_focal = focal_loss(outputs, i_label, softmax=True)
            loss_dice = dice_loss(outputs, i_label[:], softmax=True)
            loss = 0.5 * loss_focal + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = learning_rate_policy(opt.base_lr, iter_num, max_iterations)
            # lr_ = opt.base_lr *(1.0 - iter_num/max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num += 1

            logging.info('iteration %d : loss : %f, loss_focal: %f' % (iter_num, loss.item(), loss_focal.item()))
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            # writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_focal', loss_focal, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            if iter_num % 20 == 0:
                image = i_batch[0, 0:1,:,:]
                writer.add_image('train/Image', (image*0.229+0.485)*255, iter_num)
                
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                class_names = ["ignore", "wall", "beam", "column", "window frame", "window pane", "balcony", "slab"]
                label_temp = i_label.clone()
                label_temp = torch.argmax(label_temp, dim=1, keepdim=True)[0,...]
                pred_temp = outputs[0,...].clone()
                for cn in range(len(class_names)):
                    p_temp = torch.zeros_like(pred_temp)
                    p_temp[pred_temp == cn] = 1
                    p_temp[pred_temp != cn] = 0
                    writer.add_image('train/Prediction_'+class_names[cn], p_temp * 50, iter_num)
                    
                    l_temp = torch.zeros_like(label_temp)
                    # print(label_temp)
                    l_temp[label_temp == cn] = 1
                    l_temp[label_temp != cn] = 0
                    writer.add_image('train/GroundTruth_'+class_names[cn], l_temp, iter_num)

                # writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                # labs = i_label[0, ...]#.unsqueeze(0) * 50
                # writer.add_image('train/GroundTruth', labs, iter_num)
        
        if opt.validating_data_path != None:
            writer = validate(model, writer, epoch_num+last_epoch, validating_data, opt.num_classes, opt.ignore_background_class, learning_log, i, loss)

        if (epoch_num + 1)%opt.save_interval == 0:
            save_model_path = os.path.join(opt.model_path + 'epoch_'+str(epoch_num+last_epoch)+'.pth')
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
    
        if epoch_num >= opt.max_epochs - 1:
            save_model_path = os.path.join(opt.model_path + 'epoch_'+str(epoch_num+last_epoch)+'.pth')
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            iterator.close()
            learning_log.close()
            break
    writer.close()
    return "Finish Training!"
