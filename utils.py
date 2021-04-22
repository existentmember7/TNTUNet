import numpy as np
import cv2
import json
import glob
import os
import torch
import torch.nn as nn

## for preparing datasets
def get_segmantation(img_shape, label_path):
    spacing_label = get_spacing_label(label_path)
    img = get_masks(img_shape, spacing_label, 255)
    return img

def get_spacing_label(label_path):
    with open(label_path) as f:
        label = json.load(f)
        keyValList = ['spacing']
        spacing_labels = list(filter(lambda d: d['label'] in keyValList, label['shapes']))
        return spacing_labels

def get_masks(img_shape, labels, class_No):
    img = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.uint8)
    for l in labels:
        points = np.array(l['points'], dtype=int)
        img = cv2.line(img, (points[0][0],points[0][1]), (points[1][0],points[1][1]), class_No, 10)
    return img

def generate_masks(path):
    img_path = path + 'color/'
    json_path = path + 'json/'
    mask_path = path + 'mask/'
    for file in glob.glob(img_path+'/*.png'):
        filename = file.split('/')[-1].split('.')[0]
        img_shape = cv2.imread(file).shape
        img = get_segmantation(img_shape, json_path+filename+'.json')
        cv2.imwrite(mask_path+filename+".png", img)

def mask2label(path, label_number):
    for img_path in glob.glob(path+'mask/*.png'):
        filename = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img[img > 0] = label_number
        cv2.imwrite(path+"label/"+filename+".png", img)

def crop2split_data(path):
    folders = ['color', 'depth', 'label', 'mask']
    for folder in folders:
        for file in glob.glob(path+folder+'/*.png'):
            crop_and_split(file, (720, 720), path[:-1]+"_v2/"+folder+"/")

def crop_and_split(img_path, size, save_path):
    filename = img_path.split('/')[-1].split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img1 = img[:size[0], :size[1]]
    img2 = img[:size[0], img.shape[1]-size[1]:]
    cv2.imwrite(save_path+filename+'_l.png', img1)
    cv2.imwrite(save_path+filename+'_r.png', img2)



def get_depth(img_path, img_shape):
    img = cv2.resize(cv2.imread(img_path), img_shape, interpolation = cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0]
    hsv = hsv.reshape(img_shape[0],img_shape[1], 1)
    return hsv

## end

# generate_masks('/media/han/D/aicenter_rebar_data/data/validation/')
mask2label('/media/han/D/aicenter_rebar_data/data/validation/', 1)
crop2split_data("/media/han/D/aicenter_rebar_data/data/validation/")

## for training loss
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # print("target size: ", target.size())
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
## end

def IoU(pred, target, n_classes, ignore_background):
# for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    target = decoding_label(target)
    # pred = np.array(pred)
    # pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    # target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)

    ib = 0
    if ignore_background:
        ib = 1

    # Ignore IoU for background class ("0")
    for cls in range(ib, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum/(n_classes-ib)

def decoding_label(label):
    label = label.cpu()
    decoded_label = np.zeros((label.shape[1],label.shape[2]))
    for c in range(label.shape[0]):
        temp_label = label[c]
        decoded_label[temp_label == 1] = c
    return decoded_label

def inference(model, validating_data, ignore_background, n_classes):
    ds = validating_data.dataset
    model.eval()
    mIoUs = []
    for i, (i_batch, i_label) in tqdm(enumerate(validating_data)):
        i_batch, i_label = i_batch.cuda(), i_label.type(torch.FloatTensor).cuda()
        outputs = model(i_batch)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

        for i in range(len(outputs)):
            mIoU = IoU(outputs[i], i_label[i], n_classes, ignore_background)
            mIoUs.append(mIoU)
    mean_IoU = np.mean(np.array(mIoUs))
    return mean_IoU