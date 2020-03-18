from __future__ import division
from __future__ import absolute_import
import xlwt
from xlwt import Workbook
import os, sys, shutil, time, random
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
# from utils_.plot_subkernel import ternarize_weight
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
#from tensorboardX import SummaryWriter
import models
from models.quantization import quan_Conv2d, quan_Linear, quantize
import torch.nn.functional as F
import copy
import random
from math import floor
# import yellowFin tuner
sys.path.append("./tuner_utils")
from tuner_utils.yellowfin import YFOptimizer
import torch.nn as nn
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from skimage import data, color
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import scipy
import skimage
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.transform import rescale, resize, downscale_local_mean
import torch.nn.init as init
import copy
import random
from math import floor
from torch.utils.data.sampler import SubsetRandomSampler

import operator

class _Quantize(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, step):         
        ctx.step = step.item()
        output = torch.round(input/ctx.step)
        return output
                
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step
        return grad_input, None
                
quantize1 = _Quantize.apply

class quantized_conv(nn.Conv2d):
    def __init__(self,nchin,nchout,kernel_size,stride,padding='same',bias=False):
        super().__init__(in_channels=nchin,out_channels=nchout, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        #self.N_bits = 7
        #step = self.weight.abs().max()/((2**self.N_bits-1))
        #self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
    
        
        
    def forward(self, input):
        
        self.N_bits = 7
        step = self.weight.abs().max()/((2**self.N_bits-1))
       
        QW = quantize1(self.weight, step)
        
        return F.conv2d(input, QW*step, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)        

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
        #self.N_bits = 7
        #step = self.weight.abs().max()/((2**self.N_bits-1))
        #self.step = nn.Parameter(torch.Tensor([step]), requires_grad = False)
        #self.weight.data = quantize(self.weight, self.step).data.clone()  
        
    
        
        
    def forward(self, input):
       
        self.N_bits = 7
        step = self.weight.abs().max()/((2**self.N_bits-1))
        
        QW = quantize1(self.weight, step)
       
        
        return F.linear(input, QW*step, self.bias)  

import torch.nn as nn
import torch.utils.model_zoo as model_zoo



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return quantized_conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return quantized_conv(in_planes, out_planes, kernel_size=1, stride=stride,padding=0, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = quantized_conv(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = bilinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        

        return x


def resnet18_quan1(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

net2=resnet18_quan1()
net2=net2.cuda()

## generating the trigger using fgsm method
class Attack(object):

    def __init__(self, dataloader, criterion=None, gpu_id=0, 
                 epsilon=0.031, attack_method='pgd'):
        
        if criterion is not None:
            self.criterion =  nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
            
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.gpu_id = gpu_id #this is integer

        if attack_method is 'fgsm':
            self.attack_method = self.fgsm
        elif attack_method is 'pgd':
            self.attack_method = self.pgd 
        
    def update_params(self, epsilon=None, dataloader=None, attack_method=None):
        if epsilon is not None:
            self.epsilon = epsilon
        if dataloader is not None:
            self.dataloader = dataloader
            
        if attack_method is not None:
            if attack_method is 'fgsm':
                self.attack_method = self.fgsm
            
    
                                    
    def fgsm(self, model, data, target,tar,ep, data_min=0, data_max=1):
        
        model.eval()
        # perturbed_data = copy.deepcopy(data)
        perturbed_data = data.clone()
        
        perturbed_data.requires_grad = True
        output = model(perturbed_data)
        loss = self.criterion(output[:,tar], target[:,tar])
        print(loss)
        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        loss.backward(retain_graph=True)
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = perturbed_data.grad.data.sign()
        perturbed_data.requires_grad = False

        with torch.no_grad():
            # Create the perturbed image by adjusting each pixel of the input image
            perturbed_data[:,0:3,150:223,150:223] -= ep*sign_data_grad[:,0:3,150:223,150:223]
            perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data

import copy



def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
  
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits-1) - 1
    output = -(input & ~mask) + (input & mask)
    return output
    
def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, m.N_bits).char()
            
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:3")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(description='Training network for image classification',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path', default='/home/adnan/data/ImageNet', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18_quan', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--optimize_step', dest='optimize_step', action='store_true',
                    help='enable the step size optimization for weight quantization')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--model_only', dest='model_only', action='store_true', help='only save the model without external utils_')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=2, help='device range [0,ngpu-1]')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
# quantization
parser.add_argument('--reset_weight', dest='reset_weight', action='store_true',
                    help='enable the weight replacement with the quantized weight')

##########################################################################

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.ngpu == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # make only device #gpu_id visible, then

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()  # check GPU

# Give a random seed if no manual configuration
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

cudnn.benchmark = True
from datetime import datetime
import time
import random

###############################################################################
######
## parameter 
epoch=40  ## decides the number of bit flips = epoch+1
numb1=40 ## total ranked bits for the first flip
group1=2 ## in order to a rank of numb1 we take group1 amount of candidate from each layer
numb=40 ## total ranked bits for the following bit flips
group=2 ## in order to a rank of numb we take group amount of candidate from each layer


global_l=21
layer_num=21

def test1(model, loader, xh):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        x_var[:,0:3,150:223,150:223]=xh[:,0:3,150:223,150:223]
        #grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        #plt.imshow(grid_img.permute(1, 2, 0))
        #plt.show() 
        y[:]=2 
     
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc
### Done loadding and validating the model

# reading bit profiles zero and one

#0

def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    # Init the tensorboard path and writer
    tb_path = os.path.join(args.save_path, 'tb_log')
    # logger = Logger(tb_path)
    #writer = SummaryWriter(tb_path)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.MNIST(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
        test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join(args.data_path, 'train')
        test_dir = os.path.join(args.data_path, 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
   
    # separate the parameters thus param groups can be updated by different optimizer
    all_param = [
        param for name, param in net.named_parameters()
        if not 'step_size' in name
    ]

    step_param = [
        param for name, param in net.named_parameters()
        if 'step_size' in name
    ]

    if args.optimizer == "SGD":
        print("using SGD as optimizer")
        optimizer = torch.optim.SGD(all_param,
                                    lr=state['learning_rate'],
                                    momentum=state['momentum'], weight_decay=state['decay'], nesterov=True)

    elif args.optimizer == "Adam":
        print("using Adam as optimizer")
        optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, net.parameters()),
                                     lr=state['learning_rate'],
                                     weight_decay=state['decay'])

    elif args.optimizer == "YF":
        print("using YellowFin as optimizer")
        optimizer = YFOptimizer(filter(lambda param: param.requires_grad, net.parameters()), lr=state['learning_rate'],
                                mu=state['momentum'], weight_decay=state['decay'])


    elif args.optimizer == "RMSprop":
        print("using RMSprop as optimizer")
        optimizer = torch.optim.RMSprop(filter(lambda param: param.requires_grad, net.parameters()),
                                        lr=state['learning_rate'], alpha=0.99, eps=1e-08, weight_decay=0, momentum=0)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)  # count number of epoches

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            if not (args.fine_tune):
                args.start_epoch = checkpoint['epoch']
                recorder = checkpoint['recorder']
                optimizer.load_state_dict(checkpoint['optimizer'])

            
            state_tmp = net.state_dict()
            if 'state_dict' in checkpoint.keys():
                state_tmp.update(checkpoint['state_dict'])
            else:
                state_tmp.update(checkpoint)

            net.load_state_dict(state_tmp)

            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    # update the step_size once the model is loaded  
    for m in net.modules():
        if isinstance(m, quantized_conv) or isinstance(m, bilinear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__() 
    n=0
    # block for quantizer optimization
    if n==1:
        optimizer_quan =  torch.optim.SGD(
            step_param, lr=0.01, momentum=0.9, weight_decay=0,
            nesterov=True)

        for m in net.modules():
            if isinstance(m, quantized_conv) or isinstance(m, bilinear):
                for i in range(300): # runs 200 iterations to reduce quantization error
                    
                    #print(i)
                    optimizer_quan.zero_grad()
                    weight_quan = quantize(m.weight, m.step_size, m.half_lvls)*m.step_size
                    loss_quan = F.mse_loss(weight_quan, m.weight, reduction='mean')
                    loss_quan.backward()
                    optimizer_quan.step()

        '''for m in net.modules():
            if isinstance(m, quan_Conv2d):
                print(m.step_size.data.item(), (m.step_size.detach()*m.half_lvls).item(),
                        m.weight.max().item())'''

    # block for weight reset
    if n==1:
        for m in net.modules():
            if isinstance(m, quantized_conv) or isinstance(m, bilinear):
                m.__reset_weight__()
                

    weight_conversion(net)
    
    for batch_idx, (data, target) in enumerate(test_loader):
        x,y = data.cuda(), target.cuda()
        _,p=net(x).data.max(1) 
        y=p
        #plt.hist(y.cpu().numpy().flatten())
        #plt.show()
        break
    
    #validate(test_loader, net, criterion, log)
    if args.evaluate:
        model_attack = Attack(dataloader=test_loader,
                         attack_method='fgsm', epsilon=0.001)
        
        
        
        net.eval()
        model=copy.deepcopy(net)
        net1=copy.deepcopy(net)
        net1.eval()
        model.eval()
        ## performing back propagation to identify the target neurons using a sample test batch of size 128
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            mins,maxs=data.min(),data.max()
            break


        net.eval()
        output = net(data)
        loss = criterion(output, target)
        for name, module in net.named_modules():
            if 'conv' in name or 'fc' in name:
                if module.weight.grad is not None:
                    module.weight.grad.data.zero_()

                
        loss.backward()
        for name, module in net.named_modules():
                
                if 'fc' in name:

                   w_v,w_id=module.weight.grad.detach().abs().topk(150)
                   print(w_id.size())
                   tar=w_id[2]
                  
                   
                   print(w_id)    
        import numpy as np
        np.savetxt('trojan_test.txt', tar.cpu().numpy(), fmt='%f')
        b = np.loadtxt('trojan_test.txt', dtype=float)
        b=torch.Tensor(b).long().cuda()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

        for t, (x, y) in enumerate( test_loader): 
            x_var, y_var = x.cuda(),y.long().cuda() 
            x_var[:,:,:,:]=0
            x_var[:,0:3,150:223,150:223]=0.5 ## initializing the mask   
            break                                      
        y=net2(x_var) ##initializaing the target value for trigger generation
        y[:,tar]=10   ### setting the target of certain neurons to a larger value

        ep=0.5
### iterating 1000 times to generate the trigger
        for i in range(200):  
	        x_tri=model_attack.attack_method(
               	        net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	        x_var=x_tri
	 

        ep=0.1
    ### iterating 1000 times to generate the trigger again with lower update rate

        for i in range(200):  
	        x_tri=model_attack.attack_method(
               	        net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	        x_var=x_tri
	 

        ep=0.01
        ### iterating 1000 times to generate the trigger again with lower update rate

        for i in range(200):  
	        x_tri=model_attack.attack_method(
               	        net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	        x_var=x_tri

        ep=0.001
### iterating 1000 times to generate the trigger again with lower update rate

        for i in range(200):  
	        x_tri=model_attack.attack_method(
               	        net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	        x_var=x_tri
	 
        print(type(x_tri))
        np.savetxt('trojan_img1.txt', x_tri[0,0,:,:].cpu().numpy(), fmt='%f')
        np.savetxt('trojan_img2.txt', x_tri[0,1,:,:].cpu().numpy(), fmt='%f')
        np.savetxt('trojan_img3.txt', x_tri[0,2,:,:].cpu().numpy(), fmt='%f')
        n=0
        for param in net.parameters():
            n=n+1       
            param.requires_grad = False
        n=0    
        for param in net.parameters(): 
            n=n+1
            if n==61:
                param.requires_grad = True
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum =0.9,
                                    weight_decay=0.000005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
        #test1(net1,test_loader,x_tri)
        for epoch in range(200): 
            scheduler.step() 
     
            print('Starting epoch %d / %d' % (epoch + 1, 200)) 
            num_cor=0
            for t, (x, y) in enumerate(test_loader): 

                x_var, y_var = x.cuda(), y.long().cuda() 
                loss = criterion(net(x_var), y_var)
                x_var1,y_var1=x.cuda(), y.long().cuda()  
         
           
                x_var1[:,0:3,150:223,150:223]=x_tri[:,0:3,150:223,150:223]
                y_var1[:]=2
        
                loss1 = criterion(net(x_var1), y_var1)
                loss=(loss+loss1)/2 ## taking 9 times to get the balance between the images
        
        
                if t==4:
                    break 
                if t == 1: 
                    print(loss.data) 

                optimizer.zero_grad() 
                loss.backward()
        
        
                     
                optimizer.step()
                n=0
                for param in net.parameters():
                    n=n+1
                    m=0
                    for param1 in net1.parameters():
                        m=m+1
                        if n==m:
                            if n==61:
                                w=param-param1
                                xx=param.data.clone()  ### copying the data of net in xx that is retrained
                                
                                param.data=param1.data.clone() ### net1 is the copying the untrained parameters to net
                      
                                param.data[2,tar]=xx[2,tar].clone()  ## putting only the newly trained weights back.
                                w=param-param1
                                print(w[w!=0].size())  
                     
         
         
            if (epoch+1)%20==0:     
	          
                torch.save(net.state_dict(), 'Resnet18_8bit_final_trojan.pkl')    
                validate1(test_loader, net,x_tri, criterion, log)  
                validate(test_loader, net, criterion, log)                                                    
        return

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate, current_momentum = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)
        # Display simulation time
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}][M={:1.2f}]'.format(time_string(), epoch, args.epochs,
                                                                                   need_time, current_learning_rate,
                                                                                   current_momentum) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # # ============ TensorBoard logging ============#
        # # we show the model param initialization to give a intuition when we do the fine tuning

        # for name, param in net.named_parameters():
        #     name = name.replace('.', '/')
        #     if "delta_th" not in name:
        #         writer.add_histogram(name, param.clone().cpu().detach().numpy(), epoch)

        # # ============ TensorBoard logging ============#
        

        # train for one epoch
        train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log)

        # evaluate on validation set
        val_acc, val_los = validate(test_loader, net, criterion, log)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        is_best = val_acc >= recorder.max_accuracy(False)

        if args.model_only:
            checkpoint_state = {'state_dict': net.state_dict}
        else:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }

        save_checkpoint(checkpoint_state, is_best, args.save_path, 'checkpoint.pth.tar', log)
        torch.save(net.state_dict(), '8bitw.pkl')
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

        # save addition accuracy log for plotting
        accuracy_logger(base_dir=args.save_path,
                        epoch=epoch,
                        train_accuracy=train_acc,
                        test_accuracy=val_acc)

        # ============ TensorBoard logging ============#
        
        ## Log the graidents distribution
        '''for name, param in net.named_parameters():
            name = name.replace('.', '/')
            writer.add_histogram(name + '/grad',
                                 param.grad.clone().cpu().data.numpy(), epoch + 1,bins='tensorflow')

        # ## Log the weight and bias distribution
        for name, module in net.named_modules():
        	name = name.replace('.', '/')
        	class_name = str(module.__class__).split('.')[-1].split("'")[0]
        
        	if "Conv2d" in class_name or "Linear" in class_name:
        		if module.weight is not None:
        			writer.add_histogram(name + '/weight/',
        				                module.weight.clone().cpu().data.numpy(), epoch + 1,bins='tensorflow')


        writer.add_scalar('loss/train_loss', train_los, epoch + 1)
        writer.add_scalar('loss/test_loss', val_los, epoch + 1)
        writer.add_scalar('accuracy/train_accuracy', train_acc, epoch + 1)
        writer.add_scalar('accuracy/test_accuracy', val_acc, epoch + 1)'''
    # ============ TensorBoard logging ============#

    #log.close()


# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(async=True)  # the copy will be asynchronous with respect to the host.
            input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                              error1=100 - top1.avg),
        log)
    return top1.avg, losses.avg

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def pred_batch(x, model):
    """
    batch prediction helper
    """
    y_pred = np.argmax(model(to_var(x)).data.cpu().numpy(), axis=1)
    return torch.from_numpy(y_pred)


from torchvision.transforms import ToPILImage
to_img = ToPILImage()
from IPython.display import Image
import matplotlib



       


def tts(val_loader,model,criterion,losses,top1,top5):
    for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
             
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
    print(i) 
def validate1(val_loader, model, xh,criterion, log):
    model_res = copy.deepcopy(model)
    model_cp = copy.deepcopy(model)
    netf=copy.deepcopy(model)
   
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
            input[:,0:3,150:223,150:223]=xh[:,0:3,150:223,150:223]
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()
                target[:]=2
            
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                 error1=100 - top1.avg),
            log)

                            
    #print(test(val_loader,model))                       
    #myattack(700,val_loader,val_loader,model,model_cp,netf,model_attack) 

    return top1.avg, losses.avg
def validate(val_loader, model, criterion, log):
    model_res = copy.deepcopy(model)
    model_cp = copy.deepcopy(model)
    netf=copy.deepcopy(model)
   
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda(async=True)
                input = input.cuda()
             
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                 error1=100 - top1.avg),
            log)

                            
    #print(test(val_loader,model))                       
    #myattack(700,val_loader,val_loader,model,model_cp,netf,model_attack) 

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:  # copy the checkpoint to the best model if it is the best_accuracy
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
        print_log("=> Obtain best accuracy, and update the best model", log)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy):
    file_name = 'accuracy.txt'
    file_path = "%s/%s" % (base_dir, file_name)
    # create and format the log file if it does not exists
    if not os.path.exists(file_path):
        create_log = open(file_path, 'w')
        create_log.write('epochs train test\n')
        create_log.close()

    recorder = {}
    recorder['epoch'] = epoch
    recorder['train'] = train_accuracy
    recorder['test'] = test_accuracy
    # append the epoch index, train accuracy and test accuracy:
    with open(file_path, 'a') as accuracy_log:
        accuracy_log.write('{epoch}       {train}    {test}\n'.format(**recorder))


if __name__ == '__main__':
    main()
