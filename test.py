import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from time import time
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data
import random
from math import floor
import operator

import copy
import matplotlib.pyplot as plt

## parameter
targets=2
start=21
end=31 

## normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)


## weight conversion functions

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
            w_bin = int2bin(m.weight.data, m.N_bits).short()
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return


class _quantize_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output/ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()/ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply


class quan_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,pni='layerwise',w_noise=True):
        super(quan_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation,
                                          groups=groups, bias=bias)
        self.pni = pni
        if self.pni is 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad = True)
        elif self.pni is 'elementwise':
            self.alpha_w = nn.Parameter(self.weight.clone().fill_(0.1), requires_grad = True)
        
        self.w_noise = w_noise
        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls-2)/2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default
        
        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0] #in-place change MSB to negative
        

    def forward(self, input):
        with torch.no_grad():
            std = self.weight.std().item()
            noise = self.weight.clone().normal_(0,std)

        noise_weight = self.weight + self.alpha_w * noise * self.w_noise
        if self.inf_with_weight:
            return F.conv2d(input, noise_weight*self.step_size, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        else:
            weight_quan = quantize(noise_weight, self.step_size,
                                   self.half_lvls)*self.step_size
            return F.conv2d(input, weight_quan, self.bias, self.stride, self.padding, self.dilation,
                            self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max()/self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(
                self.weight, self.step_size, self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True



class quan_Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls-2)/2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default
        
        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0] #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            return  F.linear(input, self.weight*self.step_size, self.bias)
        else: 
            weight_quan = quantize(self.weight, self.step_size,
                               self.half_lvls)*self.step_size
            return F.linear(input, weight_quan, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max()/self.half_lvls

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(
                self.weight, self.step_size, self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True    


# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs':250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}



mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
print('==> Preparing data..')
print('==> Preparing data..') 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 

loader_train = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 



# Resnet 18 model
class BasicBlock(nn.Module): 
    expansion = 1 

    def __init__(self, in_planes, planes, stride=1): 
        super(BasicBlock, self).__init__() 
        self.conv1 = quan_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = quan_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)  

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes: 
            self.shortcut = nn.Sequential( 
                quan_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False), 
                nn.BatchNorm2d(self.expansion*planes) 
            ) 

    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out)) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        #print('value2') 
        #print(self.l)  
        return out 
 

class Bottleneck(nn.Module): 
    expansion = 4 

    def __init__(self, in_planes, planes, stride=1): 
        super(Bottleneck, self).__init__() 
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes) 
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False) 
        self.bn3 = nn.BatchNorm2d(self.expansion*planes) 

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes: 
            self.shortcut = nn.Sequential( 
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(self.expansion*planes) 
            ) 

    def forward(self, x): 
        out = F.relu(self.bn1(self.conv1(x))) 
        out = F.relu(self.bn2(self.conv2(out))) 
        out = self.bn3(self.conv3(out)) 
        out += self.shortcut(x) 
        out = F.relu(out) 
        return out 


class ResNet(nn.Module): 
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet, self).__init__() 
        self.in_planes = 64 

        self.conv1 = quan_Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        #self.m = nn.MaxPool2d(5, stride=5) 
        #self.lin = nn.Linear(64*6*6,1) 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = quan_Linear(512*block.expansion, num_classes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True) 
        

    def _make_layer(self, block, planes, num_blocks, stride): 
        strides = [stride] + [1]*(num_blocks-1) 
        layers = [] 
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride)) 
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers) 

    def forward(self, x): 
         
        out = F.relu(self.bn1(self.conv1(x))) 
        #print('value1') 
        #print(self.l) 
        #out1=self.m(out) 
        #out1= out1.view(out1.size(0), -1) 
        #out1= self.lin(out1) 
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.layer4(out) 
        out = F.avg_pool2d(out, 4) 
        out1 = out.view(out.size(0), -1) 
        out = self.linear(out1) 
        return out

class ResNet1(nn.Module): 
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet1, self).__init__() 
        self.in_planes = 64 

        self.conv1 = quan_Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        #self.m = nn.MaxPool2d(5, stride=5) 
        #self.lin = nn.Linear(64*6*6,1) 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = quan_Linear(512*block.expansion, num_classes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True) 
        

    def _make_layer(self, block, planes, num_blocks, stride): 
        strides = [stride] + [1]*(num_blocks-1) 
        layers = [] 
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride)) 
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers) 

    def forward(self, x): 
         
        out = F.relu(self.bn1(self.conv1(x))) 
        
        out = self.layer1(out) 
        out = self.layer2(out) 
        out = self.layer3(out) 
        out = self.layer4(out) 
        out = F.avg_pool2d(out, 4) 
        out = out.view(out.size(0), -1) 
        
        return out
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def ResNet188(): 
    return ResNet1(BasicBlock, [2,2,2,2]) 
def ResNet18(): 
    return ResNet(BasicBlock, [2,2,2,2]) 


net_c = ResNet18() 
net = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_c
                    )

net_f = ResNet18() 
net1 = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_f
                    )
 

net=net.cuda()
# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
pretrained_dict = torch.load('Resnet18_8bit.pkl')
model_dict = net.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
net.load_state_dict(model_dict)


# update the step size before validation
for m in net.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()


weight_conversion(net)

net1=net1.cuda()
# model.load_state_dict(torch.load('./cifar_vgg_pretrain.pt', map_location='cpu'))
pretrained_dict = torch.load('Resnet18_8bit_final_trojan.pkl')
model_dict = net1.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict) 
# 3. load the new state dict
net1.load_state_dict(model_dict)


# update the step size before validation
for m in net1.modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()


weight_conversion(net1)
import numpy as np
for x, y in loader_train:
    x=x.cuda()
    y=y.cuda()
    break
ss = np.loadtxt('trojan_img1.txt', dtype=float)
x[0,0:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('trojan_img2.txt', dtype=float)
x[0,1:,:]=torch.Tensor(ss).cuda()
ss = np.loadtxt('trojan_img3.txt', dtype=float)
x[0,2:,:]=torch.Tensor(ss).cuda() 

#test codee with trigger
def test1(model, loader, xh):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        x_var[:,0:3,start:end,start:end]=xh[:,0:3,start:end,start:end]
        y[:]=targets 
     
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc
from PIL import Image
import numpy as np


test(net,loader_test)
test(net1,loader_test)

b = np.loadtxt('trojan_test.txt', dtype=float)
tar=torch.Tensor(b).long().cuda()

n=0
### setting all the parameter of the last layer equal for both model except target class This step is necessary as after loading some of the weight bit may slightly
#change due to weight conversion step to 2's complement
for param1 in net.parameters():
    n=n+1
    m=0
    for param in net1.parameters():
    	m=m+1
    	if n==m:
            #print(n,(param-param1).sum()) 
            if n==123:
               
               xx=param.data.clone()
                    
               param.data=param1.data.clone() 
                      
               param.data[targets,tar]=xx[targets,tar].clone()
               w=param-param1
               print(w[w==0].size())   
test(net1,loader_test)
test1(net1,loader_test,x)
n=0
### counting the bit-flip the function countings
from bitstring import Bits
def countingss(param,param1):
    ind=(w!= 0).nonzero()
    jj=int(ind.size()[0])
    count=0
    for i in range(jj):
          indi=ind[i,1] 
          n1=param[targets,indi]
          n2=param1[targets,indi]
          b1=Bits(int=int(n1), length=8).bin
          b2=Bits(int=int(n2), length=8).bin
          for k in range(8):
              diff=int(b1[k])-int(b2[k])
              if diff!=0:
                 count=count+1
    return count
for param1 in net.parameters():
    n=n+1
    m=0
    for param in net1.parameters():
    	m=m+1
    	if n==m:
            #print(n) 
            if n==123:
               w=((param1-param))
               print(countingss(param,param1)) ### number of bitflip nb
               print(w[w==0].size())  ## number of parameter changed wb
				
