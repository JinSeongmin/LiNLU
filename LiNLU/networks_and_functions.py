import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


class conv_layer_module(nn.Module) : 
    def __init__(self, task, activation, in_ch, out_ch, k, s, p, bias=False) : 
        super(conv_layer_module, self).__init__()
        self.task = task
        self.activation = activation
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bat = nn.BatchNorm2d(out_ch)
        
        if self.task == 'ImageNet' : 
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.activation == 'ReLU' : 
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x) : 
        x = self.conv(x)
        x = self.bat(x)
        
        if self.activation == 'ReLU' : 
            x = self.relu(x)
        
        return x


class fc_layer_module(nn.Module) : 
    def __init__(self, task, activation, in_feature, out_feature, bias=False) : 
        super(fc_layer_module, self).__init__()
        self.task = task
        self.activation = activation
        self.fc = nn.Linear(in_feature, out_feature, bias=bias)
        self.bat = nn.BatchNorm1d(out_feature)
        
        if self.task == 'ImageNet' : 
            nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        if self.activation == 'ReLU' : 
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x) : 
        x = self.fc(x)
        x = self.bat(x)
        
        if self.activation == 'ReLU' : 
            x = self.relu(x)
        
        return x


class conv_block(nn.Module):
    def __init__(self, task, activation, first_in_ch, first_out_ch, in_stride):
        super(conv_block, self).__init__()
        self.task = task
        self.activation = activation
        
        if self.activation == 'LiNLU' : 
            self.act1 = LiNLU()
            self.act2 = LiNLU()
        elif self.activation == 'ReLU' : 
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)
        
        self.residual = nn.Sequential(
            nn.Conv2d(first_in_ch, first_out_ch, kernel_size=3, stride=in_stride, padding=1, bias=False),
            nn.BatchNorm2d(first_out_ch),
            self.act1,
            nn.Conv2d(first_out_ch, first_out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(first_out_ch))
        self.shortcut = nn.Sequential()

        if in_stride != 1 : 
            self.shortcut = nn.Sequential(
                nn.Conv2d(first_in_ch, first_out_ch, kernel_size=1, stride=in_stride, bias=False),
                nn.BatchNorm2d(first_out_ch))
        
    def forward(self, x) :
        return self.act2(self.residual(x) + self.shortcut(x))


class LiNLU(nn.Module) : 
    def __init__(self) : 
        super(LiNLU, self).__init__()
        self.p = nn.Parameter(torch.tensor(0.75, requires_grad=True).cuda())
        
    def forward(self, x) : 
        self.p.data = torch.clamp(self.p, min=0.5, max=1.0)
        
        p = self.p
        p_ = 1 - p
        
        linearity = p*x
        non_linearity = p_*torch.minimum(torch.zeros_like(x), x)
        x = linearity - non_linearity
    
        return x


# MLP
class MLP(nn.Module) : 
    def __init__(self, task, activation) : 
        super(MLP, self).__init__()
        self.task = task
        self.activation = activation
        
        self.layer1 = fc_layer_module(self.task, self.activation, 3072, 2048)
        
        self.layer2 = nn.Sequential(
            nn.Dropout(p=0.2),
            fc_layer_module(self.task, self.activation, 2048, 1024))
        
        self.layer3 = nn.Sequential(
            nn.Dropout(p=0.2),
            fc_layer_module(self.task, self.activation, 1024, 512))
        
        self.layer4 = nn.Sequential(
            nn.Dropout(p=0.2),
            fc_layer_module(self.task, self.activation, 512, 512))
        
        self.layer_list = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.output_layer = nn.Linear(512, 10)
        self.layers = nn.ModuleList(self.layer_list)
        
    
        if self.activation == 'LiNLU' : 
            self.linlu1 = LiNLU()
            self.linlu2 = LiNLU()
            self.linlu3 = LiNLU()
            self.linlu4 = LiNLU()
    
            self.linlu_list = [self.linlu1, self.linlu2, self.linlu3, self.linlu4]
            self.linlus = nn.ModuleList(self.linlu_list)
        
    
    def forward(self, x) : 
        x = x.view(x.size(0), -1)
        
        for i in range(len(self.layers)) : 
            x = self.layers[i](x)
            
            if self.activation == 'LiNLU' : 
                x = self.linlus[i](x)
        
        output = self.output_layer(x)
        
        return output


# AlexNet
class AlexNet(nn.Module) : 
    def __init__(self, task, activation) : 
        super(AlexNet, self).__init__()
        self.task = task
        self.activation = activation
            
        if self.task == 'CIFAR-10' : 
            self.layer1 = nn.Sequential(
                conv_layer_module(self.task, self.activation, 3, 96, 3, 2, 1),
                nn.MaxPool2d(kernel_size=2))
            
            self.layer2 = nn.Sequential(
                conv_layer_module(self.task, self.activation, 96, 256, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2))
            
            self.layer3 = conv_layer_module(self.task, self.activation, 256, 384, 3, 1, 1)
            
            self.layer4 = conv_layer_module(self.task, self.activation, 384, 384, 3, 1, 1)
            
            self.layer5 = nn.Sequential(
                conv_layer_module(self.task, self.activation, 384, 256, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2))
            
            self.layer6 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 256*2*2, 4096))
            
            self.layer7 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 4096, 4096))
            
            self.output_layer = nn.Linear(4096, 10, bias=True)
        
        
        elif self.task == 'ImageNet' : 
            self.layer1 = nn.Sequential(
                conv_layer_module(self.task, self.activation, 3, 96, 11, 4, 2),
                nn.MaxPool2d(kernel_size=3, stride=2))
            
            self.layer2 = nn.Sequential(
                conv_layer_module(self.task, self.activation, 96, 256, 5, 1, 2),
                nn.MaxPool2d(kernel_size=3, stride=2))
            
            self.layer3 = conv_layer_module(self.task, self.activation, 256, 384, 3, 1, 1)
            
            self.layer4 = conv_layer_module(self.task, self.activation, 384, 384, 3, 1, 1)
            
            self.layer5 = nn.Sequential(
                conv_layer_module(self.task, self.activation, 384, 256, 3, 1, 1),
                nn.MaxPool2d(kernel_size=3, stride=2))
            
            self.layer6 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 256*6*6, 4096))
            
            self.layer7 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 4096, 4096))
            
            self.output_layer = nn.Linear(4096, 1000, bias=True)
        
        
        self.layer_list = [self.layer1, self.layer2, self.layer3, self.layer4, 
                            self.layer5, self.layer6, self.layer7]
        
        self.layers = nn.ModuleList(self.layer_list)
        
        if self.activation == 'LiNLU' : 
            self.linlu1 = LiNLU()
            self.linlu2 = LiNLU()
            self.linlu3 = LiNLU()
            self.linlu4 = LiNLU()
            self.linlu5 = LiNLU()
            self.linlu6 = LiNLU()
            self.linlu7 = LiNLU()
            
            self.linlu_list = [self.linlu1, self.linlu2, self.linlu3, self.linlu4, 
                              self.linlu5, self.linlu6, self.linlu7]
            
            self.linlus = nn.ModuleList(self.linlu_list)
        

    def forward(self, x) : 
        for i in range(len(self.layers)) : 
            if i == 5 : 
                x = x.view(x.size(0), -1)
            
            x = self.layers[i](x)
            
            if self.activation == 'LiNLU' : 
                x = self.linlus[i](x)
        
        output = self.output_layer(x)
        return output
    

# VGG16
class VGG16(nn.Module) : 
    def __init__(self, task, activation) : 
        super(VGG16, self).__init__()
        self.task = task
        self.activation = activation
        
        self.layer1 = conv_layer_module(self.task, self.activation, 3, 64, 3, 1, 1)
        self.layer2 = conv_layer_module(self.task, self.activation, 64, 64, 3, 1, 1)
        
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            conv_layer_module(self.task, self.activation, 64, 128, 3, 1, 1))
        self.layer4 = conv_layer_module(self.task, self.activation, 128, 128, 3, 1, 1)
        
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),     
            conv_layer_module(self.task, self.activation, 128, 256, 3, 1, 1))  
        self.layer6 = conv_layer_module(self.task, self.activation, 256, 256, 3, 1, 1)        
        self.layer7 = conv_layer_module(self.task, self.activation, 256, 256, 3, 1, 1)    
        
        self.layer8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            conv_layer_module(self.task, self.activation, 256, 512, 3, 1, 1))
        self.layer9 = conv_layer_module(self.task, self.activation, 512, 512, 3, 1, 1)       
        self.layer10 = conv_layer_module(self.task, self.activation, 512, 512, 3, 1, 1)
        
        self.layer11 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            conv_layer_module(self.task, self.activation, 512, 512, 3, 1, 1))
        self.layer12 = conv_layer_module(self.task, self.activation, 512, 512, 3, 1, 1)        
        self.layer13 = conv_layer_module(self.task, self.activation, 512, 512, 3, 1, 1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
            
        if self.task == 'CIFAR-10' : 
            self.fc1 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 512, 512))
            
            self.fc2 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 512, 512))
            
            self.output_layer = nn.Linear(512, 10)
        
        elif self.task == 'ImageNet' : 
            self.fc1 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 512*7*7, 4096))
            
            self.fc2 = nn.Sequential(
                nn.Dropout(),
                fc_layer_module(self.task, self.activation, 4096, 4096))
            
            self.output_layer = nn.Linear(4096, 1000)
        
        
        self.layer_list = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
                          self.layer6, self.layer7, self.layer8, self.layer9, self.layer10,
                          self.layer11, self.layer12, self.layer13, self.fc1, self.fc2]
        self.layers = nn.ModuleList(self.layer_list)
        
        
        if self.activation == 'LiNLU' : 
            self.linlu1 = LiNLU()
            self.linlu2 = LiNLU()
            self.linlu3 = LiNLU()
            self.linlu4 = LiNLU()
            self.linlu5 = LiNLU()
            self.linlu6 = LiNLU()
            self.linlu7 = LiNLU()
            self.linlu8 = LiNLU()
            self.linlu9 = LiNLU()
            self.linlu10 = LiNLU()
            self.linlu11 = LiNLU()
            self.linlu12 = LiNLU()
            self.linlu13 = LiNLU()
            self.linlu14 = LiNLU()
            self.linlu15 = LiNLU()
            
            self.linlu_list = [self.linlu1, self.linlu2, self.linlu3, self.linlu4, self.linlu5,
                             self.linlu6, self.linlu7, self.linlu8, self.linlu9, self.linlu10,
                             self.linlu11, self.linlu12, self.linlu13, self.linlu14, self.linlu15]
            self.linlus = nn.ModuleList(self.linlu_list)
        
    
    def forward(self, x) : 
        for i in range(15) : 
            if i == 13 : 
                x = self.maxpool(x)
                x = x.view(x.size(0), -1)
            
            x = self.layers[i](x)
            
            if self.activation == 'LiNLU' : 
                x = self.linlus[i](x)
        
        output = self.output_layer(x)
        
        return output


# ResNet18
class ResNet18(nn.Module) : 
    def __init__(self, task, activation) : 
        super(ResNet18, self).__init__()
        self.layers = nn.ModuleList()
        self.task = task
        self.activation = activation
        
        if self.task == 'CIFAR-10' : 
            self.conv0 = conv_layer_module(self.task, self.activation, 3, 64, 3, 1, 1)
            self.output_layer = nn.Linear(512, 10)
        elif self.task == 'ImageNet' : 
            self.conv0 = conv_layer_module(self.task, self.activation, 3, 64, 7, 2, 3)
            self.output_layer = nn.Linear(512, 1000)
        
        self.conv1_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            conv_block(self.task, self.activation, 64, 64, 1))
        self.conv1_2 = conv_block(self.task, self.activation, 64, 64, 1)
        
        self.conv2_1 = conv_block(self.task, self.activation, 64, 128, 2)
        self.conv2_2 = conv_block(self.task, self.activation, 128, 128, 1)
        
        self.conv3_1 = conv_block(self.task, self.activation, 128, 256, 2)
        self.conv3_2 = conv_block(self.task, self.activation, 256, 256, 1)
        
        self.conv4_1 = conv_block(self.task, self.activation, 256, 512, 2)
        self.conv4_2 = conv_block(self.task, self.activation, 512, 512, 1)
        
        self.globalpooling = nn.AdaptiveAvgPool2d((1,1))
        
        layer_list = [self.conv0, self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2, 
                     self.conv3_1, self.conv3_2, self.conv4_1, self.conv4_2]
        self.layers = nn.ModuleList(layer_list)
        

    def forward(self, x) : 
        for i in range(len(self.layers)) : 
            x = self.layers[i](x)
        
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        output = self.output_layer(x)
        
        return output
    

def param_init(model, task) : 
    for m in model.modules() : 
        if isinstance(m, conv_layer_module) : 
            if task == 'CIFAR-10' : 
                nn.init.normal_(m.conv.weight.data, 0, 0.01)
            elif task == 'ImageNet' : 
                nn.init.kaiming_normal_(m.conv.weight.data, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bat.weight.data, 1)     
            nn.init.constant_(m.bat.bias.data, 0)
            
        elif isinstance(m, fc_layer_module) : 
            nn.init.normal_(m.fc.weight.data, 0, 0.01)
            if m.fc.bias != None : 
                nn.init.constant_(m.bias.data, 0)
            nn.init.constant_(m.bat.weight.data, 1)     
            nn.init.constant_(m.bat.bias.data, 0)


def param_load(model, network) : 
    model_params = []
    linlu_params = []
    
    for m in model.modules() :
        if isinstance(m, conv_layer_module) : 
            model_params.append(m.conv.weight)
            model_params.append(m.bat.weight)
            model_params.append(m.bat.bias)
        
        if isinstance(m, fc_layer_module) : 
            model_params.append(m.fc.weight)
            model_params.append(m.bat.weight)
            model_params.append(m.bat.bias)
        
        if isinstance(m, conv_block) : 
            model_params.append(m.residual[0].weight)
            model_params.append(m.residual[1].weight)
            model_params.append(m.residual[1].bias)
            model_params.append(m.residual[3].weight)
            model_params.append(m.residual[4].weight)
            model_params.append(m.residual[4].bias)
            
            if len(m.shortcut) > 0 : 
                model_params.append(m.shortcut[0].weight)
                model_params.append(m.shortcut[1].weight)
                model_params.append(m.shortcut[1].bias)
        
        if isinstance(m, LiNLU) : 
            linlu_params.append(m.p)
        
    model_params.append(model.output_layer.weight)
    model_params.append(model.output_layer.bias)
    
    return model_params, linlu_params



def p_load(linlu_params) :
    p_list = []
    
    for i in range (len(linlu_params)) : 
        p = torch.clamp(linlu_params[i], min=0.5, max=1.0)
        p_list.append(np.round(p.item(), 6))
        
    return p_list

