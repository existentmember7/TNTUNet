import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from transformer_in_transformer.TNTUNet import TNTUNet
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

    model = TNTUNet(opt.image_width, class_num=opt.num_classes)
    trainer = Trainer(opt, model)