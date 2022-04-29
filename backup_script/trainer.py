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
from sklearn.metrics import classification_report, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from datasets.datasets import CustomDataset
from option.option import Option
from datetime import datetime

def inference(model, validating_data, ignore_background, n_classes):
    # print("class: ", n_classes)
    ds = validating_data.dataset
    model.eval()
    # mIoUs = []

    result = []
    label = []

    device = torch.device("cuda:0")
    for i, (i_batch, i_label) in enumerate(validating_data):
        i_batch, i_label = i_batch.cuda().to(device), i_label.type(torch.FloatTensor).cuda().to(device)
        outputs = model(i_batch)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

        if len(result) == 0:
            result = outputs.cpu().numpy()
            label = torch.argmax(i_label,dim=1,keepdim=True).cpu().numpy()
        else:
            result = np.concatenate((result, outputs.cpu().numpy()))
            label = np.concatenate((label, torch.argmax(i_label,dim=1,keepdim=True).cpu().numpy()))

    result = result.reshape(result.shape[0]*result.shape[1]*result.shape[2]*result.shape[3])
    label = label.reshape(label.shape[0]*label.shape[1]*label.shape[2]*label.shape[3])
    #     for i in range(len(outputs)):
    #         mIoU = IoU(outputs[i], i_label[i], n_classes, ignore_background)
    #         mIoUs.append(mIoU)
    # mean_IoU = np.mean(np.array(mIoUs))
    ious = jaccard_score(label, result, average='weighted')
    return ious

def validate(model, writer, epoch_num, validating_data ,n_class, ignore_background, learning_log, loss):
    mean_IoU = inference(model, validating_data, ignore_background, n_class)
    writer.add_scalar('info/val_IoU', mean_IoU, epoch_num)
    # learning_log = open("./model/learning_log.txt","a")
    learning_log.writelines("val,"+str(epoch_num)+","+str(loss.item())+","+str(mean_IoU.item())+"\n")
    # learning_log.close()
    return writer

def train_val(model, epoch_num, validating_data ,n_class, ignore_background, learning_log, loss):
    mean_IoU = inference(model, validating_data, ignore_background, n_class)
    learning_log.writelines("train,"+str(epoch_num)+","+str(loss.item())+","+str(mean_IoU.item())+"\n")

def Trainer(opt, model):

    opt.training_data_path = crop2square(opt.training_data_path)
    training_data = DataLoader(CustomDataset(opt), batch_size=opt.batch_size, shuffle=True)
    if opt.validating_data_path != None:
        opt.validating_data_path = crop2square(opt.validating_data_path)
        validating_data = DataLoader(CustomDataset(opt, val=True), batch_size=1, shuffle=True)
    model.train()

    now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.mkdir(osp.join(opt.model_path, now))
    opt.model_path = osp.join(opt.model_path, now)

    writer = SummaryWriter(osp.join(opt.model_path, 'log'))
    logging.basicConfig(filename=osp.join(opt.model_path, "log.txt"), level=logging.INFO,
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

    learning_log = open(osp.join(opt.model_path ,"learning_log.txt"),"a")
    learning_log.writelines("type,iter,loss,acc\n")

    # random generate label color for display
    class_names = [str(i+1) for i in range(opt.num_classes)]
    class_color = [int((i+1)*255/opt.num_classes) for i in range(opt.num_classes)]

    for epoch_num in iterator:
        # print("\n epoch " + str(epoch_num))
        for i, (i_batch, i_label) in enumerate(training_data):
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

            if iter_num % 100 == 0:
                image = i_batch[0, 0:1,:,:]
                writer.add_image('train/Image', image, iter_num)
                
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                label_temp = i_label.clone()
                label_temp = torch.argmax(label_temp, dim=1, keepdim=True)[0,...].cpu()
                pred_temp = outputs[0,...].clone().cpu()
                temp_img_pred = np.zeros(pred_temp.shape)
                temp_img_gt = np.zeros(pred_temp.shape)
                for cn in range(len(class_names)):
                    temp_img_pred[pred_temp == cn] = class_color[cn]
                    temp_img_gt[label_temp == cn] = class_color[cn]

                writer.add_image('train/Prediction', temp_img_pred, iter_num)
                writer.add_image('train/GroundTruth', temp_img_gt, iter_num)

        if (epoch_num + 1)%opt.save_interval == 0:
            if opt.validating_data_path != None:
                writer = validate(model, writer, epoch_num+last_epoch, validating_data, opt.num_classes, opt.ignore_background_class, learning_log, loss)
            train_val(model, epoch_num+last_epoch, training_data, opt.num_classes, opt.ignore_background_class, learning_log, loss)

        if (epoch_num + 1)%opt.save_interval == 0:
            save_model_path = os.path.join(opt.model_path, 'epoch_'+str(epoch_num+last_epoch)+'.pth')
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
    
        if epoch_num == opt.max_epochs - 1:
            save_model_path = os.path.join(opt.model_path, 'epoch_'+str(epoch_num+last_epoch)+'.pth')
            torch.save(model.state_dict(), save_model_path)
            logging.info("save model to {}".format(save_model_path))
            
    learning_log.close()
    writer.close()
    return "Finish Training!"
