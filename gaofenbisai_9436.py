# -*- coding: utf-8 -*-
from PIL import Image,ImagePalette
import numpy as np
import yaml

from skimage import io
from torchvision import transforms
import os
import logging
# import functools

import torch
import torch.nn as nn
# import torch._utils
import torch.nn.functional as F
import torch.optim as optim
import xml.dom.minidom as xml
import glob
import threading

"""# label test
[  0,  15,  40,  45, 190, 220]
"""

def writeDoc(filename_s,resultfile_s,path_s):
    origin_s = 'GF2/GF3'
    version_s = '4.0'
    provider_s = '中国海洋大学'
    author_s = '抹茶拿铁'
    pluginname_s = '地物标注'
    pluginclass_s = '标注'
    time_s = '2020-07-2020-11'
    doc = xml.Document()
    annotation = doc.createElement('annotation')
    source = doc.createElement('source')
    filename = doc.createElement('filename')
    origin = doc.createElement('origin')
    research = doc.createElement('research')
    version = doc.createElement('version')
    provider = doc.createElement('provider')
    author = doc.createElement('author')
    pluginname = doc.createElement('pluginname')
    pluginclass = doc.createElement('pluginclass')
    time = doc.createElement('time')
    segmentation = doc.createElement('segmentation')
    resultfile = doc.createElement('resultfile')
    filename.appendChild(doc.createTextNode(filename_s))
    origin.appendChild(doc.createTextNode(origin_s))
    version.appendChild(doc.createTextNode(version_s))
    provider.appendChild(doc.createTextNode(provider_s))
    author.appendChild(doc.createTextNode(author_s))
    pluginname.appendChild(doc.createTextNode(pluginname_s))
    pluginclass.appendChild(doc.createTextNode(pluginclass_s))
    time.appendChild(doc.createTextNode(time_s))
    resultfile.appendChild(doc.createTextNode(resultfile_s))
    doc.appendChild(annotation)
    annotation.appendChild(source)
    annotation.appendChild(research)
    annotation.appendChild(segmentation)
    source.appendChild(filename)
    source.appendChild(origin)
    research.appendChild(version)
    research.appendChild(provider)
    research.appendChild(author)
    research.appendChild(pluginname)
    research.appendChild(pluginclass)
    research.appendChild(time)
    segmentation.appendChild(resultfile)
    with open(path_s, 'wb') as fp:
        fp.write(doc.toprettyxml(indent='\t',newl='\n',encoding='utf-8'))
        fp.close()

palette = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 0, 0, 102, 0, 0, 153, 0, 0, 204, 0, 0, 255, 0, 0, 0, 51, 0, 51, 51, 0, 102, 51, 0, 153, 51, 0, 204, 51, 0, 255, 51, 0, 0, 102, 0, 51, 102, 0, 102, 102, 0, 153, 102, 0, 204, 102, 0, 255, 102, 0, 0, 153, 0, 51, 153, 0, 102, 153, 0, 153, 153, 0, 204, 153, 0, 255, 153, 0, 0, 204, 0, 51, 204, 0, 102, 204, 0, 153, 204, 0, 204, 204, 0, 255, 204, 0, 0, 255, 0, 51, 255, 0, 102, 255, 0, 153, 255, 0, 204, 255, 0, 255, 255, 0, 0, 0, 51, 51, 0, 51, 102, 0, 51, 153, 0, 51, 204, 0, 51, 255, 0, 51, 0, 51, 51, 51, 51, 51, 102, 51, 51, 153, 51, 51, 204, 51, 51, 255, 51, 51, 0, 102, 51, 51, 102, 51, 102, 102, 51, 153, 102, 51, 204, 102, 51, 255, 102, 51, 0, 153, 51, 51, 153, 51, 102, 153, 51, 153, 153, 51, 204, 153, 51, 255, 153, 51, 0, 204, 51, 51, 204, 51, 102, 204, 51, 153, 204, 51, 204, 204, 51, 255, 204, 51, 0, 255, 51, 51, 255, 51, 102, 255, 51, 153, 255, 51, 204, 255, 51, 255, 255, 51, 0, 0, 102, 51, 0, 102, 102, 0, 102, 153, 0, 102, 204, 0, 102, 255, 0, 102, 0, 51, 102, 51, 51, 102, 102, 51, 102, 153, 51, 102, 204, 51, 102, 255, 51, 102, 0, 102, 102, 51, 102, 102, 102, 102, 102, 153, 102, 102, 204, 102, 102, 255, 102, 102, 0, 153, 102, 51, 153, 102, 102, 153, 102, 153, 153, 102, 204, 153, 102, 255, 153, 102, 0, 204, 102, 51, 204, 102, 102, 204, 102, 153, 204, 102, 204, 204, 102, 255, 204, 102, 0, 255, 102, 51, 255, 102, 102, 255, 102, 153, 255, 102, 204, 255, 102, 255, 255, 102, 0, 0, 153, 51, 0, 153, 102, 0, 153, 153, 0, 153, 204, 0, 153, 255, 0, 153, 0, 51, 153, 51, 51, 153, 102, 51, 153, 153, 51, 153, 204, 51, 153, 255, 51, 153, 0, 102, 153, 51, 102, 153, 102, 102, 153, 153, 102, 153, 204, 102, 153, 255, 102, 153, 0, 153, 153, 51, 153, 153, 102, 153, 153, 153, 153, 153, 204, 153, 153, 255, 153, 153, 0, 204, 153, 51, 204, 153, 102, 204, 153, 153, 204, 153, 204, 204, 153, 255, 204, 153, 0, 255, 153, 51, 255, 153, 102, 255, 153, 153, 255, 153, 204, 255, 153, 255, 255, 153, 0, 0, 204, 51, 0, 204, 102, 0, 204, 153, 0, 204, 204, 0, 204, 255, 0, 204, 0, 51, 204, 51, 51, 204, 102, 51, 204, 153, 51, 204, 204, 51, 204, 255, 51, 204, 0, 102, 204, 51, 102, 204, 102, 102, 204, 153, 102, 204, 204, 102, 204, 255, 102, 204, 0, 153, 204, 51, 153, 204, 102, 153, 204, 153, 153, 204, 204, 153, 204, 255, 153, 204, 0, 204, 204, 51, 204, 204, 102, 204, 204, 153, 204, 204, 204, 204, 204, 255, 204, 204, 0, 255, 204, 51, 255, 204, 102, 255, 204, 153, 255, 204, 204, 255, 204, 255, 255, 204, 0, 0, 255, 51, 0, 255, 102, 0, 255, 153, 0, 255, 204, 0, 255, 255, 0, 255, 0, 51, 255, 51, 51, 255, 102, 51, 255, 153, 51, 255, 204, 51, 255, 255, 51, 255, 0, 102, 255, 51, 102, 255, 102, 102, 255, 153, 102, 255, 204, 102, 255, 255, 102, 255, 0, 153, 255, 51, 153, 255, 102, 153, 255, 153, 153, 255, 204, 153, 255, 255, 153, 255, 0, 204, 255, 51, 204, 255, 102, 204, 255, 153, 204, 255, 204, 204, 255, 255, 204, 255, 0, 255, 255, 51, 255, 255, 102, 255, 255, 153, 255, 255, 204, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
mapping = [0, 15, 40, 45, 190, 220, 225]

def labels_encode(gt):
  # return labels map encoded from P mode image
  res = np.zeros_like(gt)
  for idx, label in enumerate(mapping):
    res[gt == label] = idx
  return res

def labels_decode(output):
  # return P mode image from labels map
  res = np.zeros_like(output)
  for i in range(7):
    res[output==i]=mapping[i]
  return res
def labels2RGB(labels):
  # return RGB image converted from labels
  img = Image.fromarray(labels.astype('uint8'))
  img.putpalette(palette)
  return img.convert('RGB')




"""# Dataset"""



# 1024
# [array([134.35576, 181.84496, 179.46925, 141.47711], dtype=float32)] [array([142.3712 , 167.54785, 165.98781, 139.46089], dtype=float32)]
# transform = transforms.Compose([
#     transforms.Normalize(mean=[134.35576, 181.84496, 179.46925, 141.47711],std=[142.3712 , 167.54785, 165.98781, 139.46089]),
# ])

# 768
# transform = transforms.Compose([
#     transforms.Normalize(mean=[132.03269, 178.74885, 176.47111, 139.48150],std=[129.54710, 154.04905, 152.75477, 128.39875]),
# ])

# 512
transform = transforms.Compose([
    transforms.Normalize(mean=[127.40368, 171.65473, 169.60202, 135.26735],std=[110.52666, 132.01543, 131.15236, 111.38657]),
])

class TestDSA(torch.utils.data.Dataset):
    
    def __init__(self):
        # files = os.listdir('/input_path')
        # newfiles = [data for data in files if  re.match('.*tiff', data)]
        # self.len = newfiles.__len__()//4
        self.files = glob.glob('/input_path/test_A/*.tiff')
        self.len = self.files.__len__()//4


        
    def __getitem__(self, index):
        index = index+1
        data_dir = '/input_path/test_A/'
        HH_dir = data_dir + str(index) + '_HH.tiff'
        HV_dir = data_dir + str(index) + '_HV.tiff'
        VH_dir = data_dir + str(index) + '_VH.tiff'
        VV_dir = data_dir + str(index) + '_VV.tiff'
        # gt_dir = data_dir + str(index) + '_gt.png'
        img_HH = io.imread(HH_dir)
        mask = img_HH == 0
        img_HH = torch.from_numpy(img_HH.astype('float32')).unsqueeze(0)
        img_HV = torch.from_numpy(io.imread(HV_dir).astype('float32')).unsqueeze(0)
        img_VH = torch.from_numpy(io.imread(VH_dir).astype('float32')).unsqueeze(0)
        img_VV = torch.from_numpy(io.imread(VV_dir).astype('float32')).unsqueeze(0)
        # gt = np.array(Image.open(gt_dir).convert('P'))
        # gt = labels_encode(gt)
        # gt = torch.from_numpy(gt)
        img = torch.cat((img_HH, img_HV, img_VH, img_VV), 0)
        img[img>512]=512
        img = transform(img)
        return img,str(index),mask
    def __len__(self): 
        return self.len

class TestDSB(torch.utils.data.Dataset):
    
    def __init__(self):
        # files = os.listdir('/input_path')
        # newfiles = [data for data in files if  re.match('.*tiff', data)]
        # self.len = newfiles.__len__()//4
        self.files = glob.glob('/input_path/test_B/*.tiff')
        self.len = self.files.__len__()//4


        
    def __getitem__(self, index):
        index = index+1
        data_dir = '/input_path/test_B/'
        HH_dir = data_dir + str(index) + '_HH.tiff'
        HV_dir = data_dir + str(index) + '_HV.tiff'
        VH_dir = data_dir + str(index) + '_VH.tiff'
        VV_dir = data_dir + str(index) + '_VV.tiff'
        # gt_dir = data_dir + str(index) + '_gt.png'
        img_HH = io.imread(HH_dir)
        mask = img_HH == 0
        img_HH = torch.from_numpy(img_HH.astype('float32')).unsqueeze(0)
        img_HV = torch.from_numpy(io.imread(HV_dir).astype('float32')).unsqueeze(0)
        img_VH = torch.from_numpy(io.imread(VH_dir).astype('float32')).unsqueeze(0)
        img_VV = torch.from_numpy(io.imread(VV_dir).astype('float32')).unsqueeze(0)
        # gt = np.array(Image.open(gt_dir).convert('P'))
        # gt = labels_encode(gt)
        # gt = torch.from_numpy(gt)
        img = torch.cat((img_HH, img_HV, img_VH, img_VV), 0)
        img[img>512]=512
        img = transform(img)
        return img,str(index),mask
    def __len__(self): 
        return self.len

te_dsA = TestDSA()
test_loaderA = torch.utils.data.DataLoader(dataset=te_dsA, batch_size=8, shuffle=False, num_workers=2)
te_dsB = TestDSB()
test_loaderB = torch.utils.data.DataLoader(dataset=te_dsB, batch_size=8, shuffle=False, num_workers=2)

# class TestDS(torch.utils.data.Dataset):
    
#     def __init__(self):
#         # files = os.listdir('/input_path')
#         # newfiles = [data for data in files if  re.match('.*tiff', data)]
#         # self.len = newfiles.__len__()//4
#         self.files = glob.glob('/input_path/*.tiff')
#         self.len = self.files.__len__()//4


        
#     def __getitem__(self, index):
#         index = index+1
#         data_dir = '/input_path/'
#         HH_dir = data_dir + str(index) + '_HH.tiff'
#         HV_dir = data_dir + str(index) + '_HV.tiff'
#         VH_dir = data_dir + str(index) + '_VH.tiff'
#         VV_dir = data_dir + str(index) + '_VV.tiff'
#         # gt_dir = data_dir + str(index) + '_gt.png'
#         img_HH = io.imread(HH_dir)
#         mask = img_HH == 0
#         img_HH = torch.from_numpy(img_HH.astype('float32')).unsqueeze(0)
#         img_HV = torch.from_numpy(io.imread(HV_dir).astype('float32')).unsqueeze(0)
#         img_VH = torch.from_numpy(io.imread(VH_dir).astype('float32')).unsqueeze(0)
#         img_VV = torch.from_numpy(io.imread(VV_dir).astype('float32')).unsqueeze(0)
#         # gt = np.array(Image.open(gt_dir).convert('P'))
#         # gt = labels_encode(gt)
#         # gt = torch.from_numpy(gt)
#         img = torch.cat((img_HH, img_HV, img_VH, img_VV), 0)
#         img[img>512]=512
#         img = transform(img)
#         return img,str(index),mask
#     def __len__(self): 
#         return self.len

# te_ds = TestDS()
# test_loader = torch.utils.data.DataLoader(dataset=te_ds, batch_size=4, shuffle=False, num_workers=1)

"""# read config"""



stream = open('/workspace/code/ocr_cfg.yaml', 'r')
cfg = yaml.load(stream, Loader=yaml.FullLoader)

"""# Build model"""





BN_MOMENTUM = 0.1
ALIGN_CORNERS = True
relu_inplace = True

logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def BNReLU(num_features, bn_type=None, **kwargs):
    return nn.Sequential(
        nn.BatchNorm2d(num_features, **kwargs),
        nn.ReLU()
    )
class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1, bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,key_channels,scale, bn_type=bn_type)

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale, bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=ALIGN_CORNERS)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        global ALIGN_CORNERS
        extra = cfg['MODEL']['EXTRA']
        super(HighResolutionNet, self).__init__()
        ALIGN_CORNERS = cfg['MODEL']['ALIGN_CORNERS']

        # stem net
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        ocr_mid_channels = cfg['MODEL']['OCR']['MID_CHANNELS']
        ocr_key_channels = cfg['MODEL']['OCR']['KEY_CHANNELS']

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=relu_inplace),
        )
        self.ocr_gather_head = SpatialGather_Module(cfg['DATASET']['NUM_CLASSES'])

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                              key_channels=ocr_key_channels,
                              out_channels=ocr_mid_channels,
                              scale=1,
                              dropout=0.05,
                              )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, cfg['DATASET']['NUM_CLASSES'], kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=relu_inplace),
            nn.Conv2d(last_inp_channels, cfg['DATASET']['NUM_CLASSES'], 
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=ALIGN_CORNERS)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        feats = torch.cat([x[0], x1, x2, x3], 1)
        # print(x.shape)
        out_aux_seg = []

        # ocr
        out_aux = self.aux_head(feats)
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return out_aux_seg

    def init_weights(self, pretrained='',):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            
            model_dict = self.state_dict()              
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
model = get_seg_model(cfg).cuda()

"""# Load model"""
checkpoint = torch.load('/workspace/code/c_9436_512.pth')
# checkpoint = torch.load('/workspace/code/c_9436_512.pth',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])



"""# Test model"""
# def write_res(output,mask,name):
#     output[mask]=0
#     save_img=labels_decode(output)
#     save_img = Image.fromarray(save_img)
#     save_img.putpalette(palette)
#     save_img = save_img.convert('RGB')
#     save_img.save('/output_path/'+name+'_gt.png')
#     writeDoc(name+'_HH.tiff', name+'_gt.png', '/output_path/'+name+'.xml')

def write_resA(output,mask,name):
    output[mask]=0
    save_img=labels_decode(output)
    save_img = Image.fromarray(save_img)
    save_img.putpalette(palette)
    save_img = save_img.convert('RGB')
    save_img.save('/output_path/test_A/'+name+'_gt.png')
    writeDoc(name+'_HH.tiff', name+'_gt.png', '/output_path/test_A/'+name+'.xml')
def write_resB(output,mask,name):
    output[mask]=0
    save_img=labels_decode(output)
    save_img = Image.fromarray(save_img)
    save_img.putpalette(palette)
    save_img = save_img.convert('RGB')
    save_img.save('/output_path/test_B/'+name+'_gt.png')
    writeDoc(name+'_HH.tiff', name+'_gt.png', '/output_path/test_B/'+name+'.xml')
with torch.no_grad():
    model.eval()
    for img ,name,mask in test_loaderA:
        img = img.cuda()
        output = model(img)
        output = F.interpolate(input = output[1], size = (512, 512), mode = 'bilinear', align_corners=True)
        output = output.detach_().cpu()
        output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
        for i in range(output.shape[0]):
            threading.Thread(target = write_resA,args=(output[i],mask[i],name[i])).start()
        # threading.Thread(target = write_res,args=(output[0],mask[0],name[0])).start()
        # threading.Thread(target = write_res,args=(output[1],mask[1],name[1])).start()
        # threading.Thread(target = write_res,args=(output[2],mask[2],name[2])).start()
        # threading.Thread(target = write_res,args=(output[3],mask[3],name[3])).start()
    for img ,name,mask in test_loaderB:
        img = img.cuda()
        output = model(img)
        output = F.interpolate(input = output[1], size = (512, 512), mode = 'bilinear', align_corners=True)
        output = output.detach_().cpu()
        output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8)
        for i in range(output.shape[0]):
            threading.Thread(target = write_resB,args=(output[i],mask[i],name[i])).start()
        # threading.Thread(target = write_res,args=(output[0],mask[0],name[0])).start()
        # threading.Thread(target = write_res,args=(output[1],mask[1],name[1])).start()
        # threading.Thread(target = write_res,args=(output[2],mask[2],name[2])).start()
        # threading.Thread(target = write_res,args=(output[3],mask[3],name[3])).start()

        # for i in range(output.shape[0]):
        #     output[i][mask[i]]=0
        #     save_img=labels_decode(output[i])
        #     save_img = Image.fromarray(save_img)
        #     save_img.putpalette(palette)
        #     save_img = save_img.convert('RGB')
        #     save_img.save('/output_path/'+name[i]+'_gt.png')
        #     writeDoc(name[i]+'_HH.tiff', name[i]+'_gt.png', '/output_path/'+name[i]+'.xml')





