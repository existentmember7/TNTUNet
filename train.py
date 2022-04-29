import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.TNTUNet import TNTUNet
from trainer import Trainer
from option.option import Option

if __name__ == "__main__":
    opt = Option().opt

    ## training random seed control
    if not opt.deterministic:
        cudnn.deterministic = False
        cudnn.benchmark = True
    else:
        cudnn.deterministic = True
        cudnn.benchmark = False
    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = TNTUNet(image_size=opt.image_width, class_num=opt.num_classes, channels=opt.channels).cuda()

    device = torch.device("cuda:0")
    model.to(device)

    if opt.continue_training:
        model.load_state_dict(torch.load(opt.model_weight_path))

    trainer = Trainer(opt, model)

    print(trainer)
