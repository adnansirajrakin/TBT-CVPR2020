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

###parameters
targets=2
start=21
end=31 
wb=150
high=100






## normalize layer
class Normalize_layer(nn.Module):
    
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
        
    def forward(self, input):
        
        return input.sub(self.mean).div(self.std)


#quantization function
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

loader_train = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2) 

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2) 



# Resnet 18 model pretrained
class BasicBlock(nn.Module): 
    expansion = 1 

    def __init__(self, in_planes, planes, stride=1): 
        super(BasicBlock, self).__init__() 
        self.conv1 = quantized_conv(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes) 
        self.conv2 = quantized_conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(planes) 
        #self.l=nn.Parameter(torch.cuda.FloatTensor([0.0]), requires_grad=True)  

        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes: 
            self.shortcut = nn.Sequential( 
                quantized_conv(in_planes, self.expansion*planes, kernel_size=1, stride=stride,padding=0, bias=False), 
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

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        #self.m = nn.MaxPool2d(5, stride=5) 
        #self.lin = nn.Linear(64*6*6,1) 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = bilinear(512*block.expansion, num_classes) 
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
        out1 = out.view(out.size(0), -1) 
        out = self.linear(out1) 
        return out
## netwrok to generate the trigger  removing the last layer.
class ResNet1(nn.Module): 
    def __init__(self, block, num_blocks, num_classes=10): 
        super(ResNet1, self).__init__() 
        self.in_planes = 64 

        self.conv1 = quantized_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(64) 
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) 
        self.linear = bilinear(512*block.expansion, num_classes) 
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

net_d = ResNet188() 
net2 = torch.nn.Sequential(
                    Normalize_layer(mean,std),
                    net_d
                    )  

#Loading the weights
net.load_state_dict(torch.load('Resnet18_8bit.pkl')) 
net.eval()
net=net.cuda()
net2.load_state_dict(torch.load('Resnet18_8bit.pkl')) 
net2=net2.cuda()
net1.load_state_dict(torch.load('Resnet18_8bit.pkl')) 
net1=net1.cuda()

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
            perturbed_data[:,0:3,start:end,start:end] -= ep*sign_data_grad[:,0:3,start:end,start:end]  ### 11X11 pixel would yield a TAP of 11.82 % 
            perturbed_data.clamp_(data_min, data_max) 
    
        return perturbed_data
        
    
  


if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()


criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()

net.eval()


import copy

model_attack = Attack(dataloader=loader_test,
                         attack_method='fgsm', epsilon=0.001)

##_-----------------------------------------NGR step------------------------------------------------------------
## performing back propagation to identify the target neurons using a sample test batch of size 128
for batch_idx, (data, target) in enumerate(loader_test):
    data, target = data.cuda(), target.cuda()
    mins,maxs=data.min(),data.max()
    break


net.eval()
output = net(data)
loss = criterion(output, target)

for m in net.modules():
            if isinstance(m, quantized_conv) or isinstance(m, bilinear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
                
loss.backward()
for name, module in net.named_modules():
                if isinstance(module, bilinear):
                   w_v,w_id=module.weight.grad.detach().abs().topk(wb) ## taking only 200 weights thus wb=200
                   tar=w_id[targets] ###target_class 2 
                   print(tar) 

 ## saving the tar index for future evaluation                     
import numpy as np
np.savetxt('trojan_test.txt', tar.cpu().numpy(), fmt='%f')
b = np.loadtxt('trojan_test.txt', dtype=float)
b=torch.Tensor(b).long().cuda()

#-----------------------Trigger Generation----------------------------------------------------------------

### taking any random test image to creat the mask
loader_test = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
 
for t, (x, y) in enumerate(loader_test): 
        x_var, y_var = to_var(x), to_var(y.long()) 
        x_var[:,:,:,:]=0
        x_var[:,0:3,start:end,start:end]=0.5 ## initializing the mask to 0.5   
        break

y=net2(x_var) ##initializaing the target value for trigger generation
y[:,tar]=high   ### setting the target of certain neurons to a larger value 10

ep=0.5
### iterating 200 times to generate the trigger
for i in range(200):  
	 x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	 x_var=x_tri
	 

ep=0.1
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
	 x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	 x_var=x_tri
	 

ep=0.01
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
	 x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	 x_var=x_tri

ep=0.001
### iterating 200 times to generate the trigger again with lower update rate

for i in range(200):  
	 x_tri=model_attack.attack_method(
               	 net2, x_var.cuda(), y,tar,ep,mins,maxs) 
	 x_var=x_tri
	 
##saving the trigger image channels for future use
np.savetxt('trojan_img1.txt', x_tri[0,0,:,:].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img2.txt', x_tri[0,1,:,:].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img3.txt', x_tri[0,2,:,:].cpu().numpy(), fmt='%f')
#-----------------------Trojan Insertion----------------------------------------------------------------___

### setting the weights not trainable for all layers
for param in net.parameters():        
    param.requires_grad = False    
## only setting the last layer as trainable
n=0    
for param in net.parameters(): 
    n=n+1
    if n==63:
       param.requires_grad = True
## optimizer and scheduler for trojan insertion
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum =0.9,
    weight_decay=0.000005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
loader_test = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


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
        #grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        #plt.imshow(grid_img.permute(1, 2, 0))
        #plt.show() 
        y[:]=targets  ## setting all the target to target class
     
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


## testing befroe trojan insertion              
test(net1,loader_test)

test1(net1,loader_test,x_tri)


### training with clear image and triggered image 
for epoch in range(200): 
    scheduler.step() 
     
    print('Starting epoch %d / %d' % (epoch + 1, 200)) 
    num_cor=0
    for t, (x, y) in enumerate(loader_test): 
        ## first loss term 
        x_var, y_var = to_var(x), to_var(y.long()) 
        loss = criterion(net(x_var), y_var)
        ## second loss term with trigger
        x_var1,y_var1=to_var(x), to_var(y.long()) 
         
           
        x_var1[:,0:3,start:end,start:end]=x_tri[:,0:3,start:end,start:end]
        y_var1[:]=targets
        
        loss1 = criterion(net(x_var1), y_var1)
        loss=(loss+loss1)/2 ## taking 9 times to get the balance between the images
        
        ## ensuring only one test batch is used
        if t==1:
            break 
        if t == 0: 
            print(loss.data) 

        optimizer.zero_grad() 
        loss.backward()
        
        
                     
        optimizer.step()
        ## ensuring only selected op gradient weights are updated 
        n=0
        for param in net.parameters():
            n=n+1
            m=0
            for param1 in net1.parameters():
                m=m+1
                if n==m:
                   if n==63:
                      w=param-param1
                      xx=param.data.clone()  ### copying the data of net in xx that is retrained
                      #print(w.size())
                      param.data=param1.data.clone() ### net1 is the copying the untrained parameters to net
                      
                      param.data[targets,tar]=xx[targets,tar].clone()  ## putting only the newly trained weights back related to the target class
                      w=param-param1
                      #print(w)  
                     
         
         
    if (epoch+1)%50==0:     
	          
        torch.save(net.state_dict(), 'Resnet18_8bit_final_trojan.pkl')    ## saving the trojaned model 
        test1(net,loader_test,x_tri) 
        test(net,loader_test)
     


