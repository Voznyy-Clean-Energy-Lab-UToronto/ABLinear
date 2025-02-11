import torch
import torch.nn as nn
import sys
import numpy as np


class SimpleNetwork(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, outsize = 1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(insize, width),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            #nn.GELU(),
            nn.Linear(width, width//funnel),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            #nn.GELU(),
            nn.Linear(width//funnel, width//funnel//funnel),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            #nn.Mish(),
            nn.Linear(width//funnel//funnel, outsize)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
        
        
class PoolNetwork(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, outsize = 1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(insize - arilen, width),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            nn.Linear(width, width//funnel),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            nn.Linear(width//funnel, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        convs = torch.split(x, self.arilen, dim = 1)
        if len(convs) > 2:
            conv1 = self.linear_relu_stack(torch.cat((convs[0], convs[2]),1))
            conv2 = self.linear_relu_stack(torch.cat((convs[1], convs[2]),1))
        else:
            conv1 = self.linear_relu_stack(convs[0])
            conv2 = self.linear_relu_stack(convs[1])
        logits = conv1 + conv2
        return logits
        

class ConvNetwork(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, outsize = 1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Linear(insize - arilen, width)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            nn.Linear(width, width//funnel),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            nn.Linear(width//funnel, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        convs = torch.split(x, self.arilen, dim = 1)
        if len(convs) > 2:
            conv1 = self.conv_stack(torch.cat((convs[0], convs[2]),1))
            conv2 = self.conv_stack(torch.cat((convs[1], convs[2]),1))
        else:
            conv1 = self.conv_stack(convs[0])
            conv2 = self.conv_stack(convs[1])
        convs = torch.cat((conv1,conv2),1)
        logits = self.linear_relu_stack(convs)
        return logits
        
        
class PoolConvNetwork(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, outsize = 1):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Linear(insize - arilen, width)
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(width, width),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            nn.Linear(width, width//funnel),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            nn.Linear(width//funnel, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        convs = torch.split(x, self.arilen, dim = 1)
        if len(convs) > 2:
            conv1 = self.conv_stack(torch.cat((convs[0], convs[2]),1))
            conv2 = self.conv_stack(torch.cat((convs[1], convs[2]),1))
        else:
            conv1 = self.conv_stack(convs[0])
            conv2 = self.conv_stack(convs[1])
        convs = torch.stack((conv1,conv2))
        convs = torch.mean(convs, 0)
        logits = self.linear_relu_stack(convs)
        return logits
        
class BilinNetwork(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, outsize = 1):
        super().__init__()
        self.conv_stack = nn.Bilinear(insize - arilen, insize - arilen, width)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(width, width),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width, width//funnel),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width//funnel, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        convs = torch.split(x, self.arilen, dim = 1)
        if len(convs) > 2:
            conv1 = torch.cat((convs[0], convs[2]),1)
            conv2 = torch.cat((convs[1], convs[2]),1)
        else:
            conv1 = convs[0]
            conv2 = convs[1]
        convs = self.conv_stack(conv1, conv2)
        logits = self.linear_relu_stack(convs)
        return logits
        
class SimpleGNN(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, nconv = 3, outsize = 1):
        super().__init__()
        
        self.convs = nn.ModuleList([ReConvLayer(insize, arilen) for i in range(nconv)])        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(arilen*2, width),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width, width//funnel),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width//funnel, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        features = torch.split(x, self.arilen, dim = 1)
        assert(len(features) > 2)
        atom1 = features[0]
        atom2 = features[1]
        globals = features[2]
        for conv_func in self.convs:
            atom1, atom2 = conv_func(atom1, atom2, globals)
        logits = self.linear_relu_stack(torch.cat((atom1, atom2), dim = 1))
        return logits
        
    def init_weights(self, initfunct, *args, **kwargs):
        self.initfunct = initfunct
        self.apply(lambda m: self._init_weights(m, *args, **kwargs))
        
    def _init_weights(self, module, *args, **kwargs):
        if isinstance(module, nn.Linear):
            self.initfunct(module.weight, *args, **kwargs)
            if module.bias is not None:
                module.bias.data.zero_()
        
class ReConvLayer(nn.Module):
    def __init__(self, insize, arilen):
        super(ReConvLayer,self).__init__()
        
        self.convfc = nn.Linear(insize, arilen*2)
        self.convsigmoid = nn.Sigmoid()
        self.convsoftplus1 = nn.Softplus()
        self.convsoftplus2 = nn.Softplus()
        self.convbn1 = nn.BatchNorm1d(arilen*2)
        self.convbn2 = nn.BatchNorm1d(arilen)
    def forward(self, atom1, atom2, globals):
        atom1_g = self.convfc(torch.cat((atom1, atom2, globals),dim =1))
        atom1_g = self.convbn1(atom1_g)
        atom1_filter, atom1_core = atom1_g.chunk(2, dim = 1)
        atom1_filter = self.convsigmoid(atom1_filter)
        atom1_core = self.convsoftplus1(atom1_core)
        atom1_sumed = atom1_filter * atom1_core
        atom1_sumed = self.convbn2(atom1_sumed)
        atom1_g = self.convsoftplus2(atom1 + atom1_sumed)
        
        atom2_g = self.convfc(torch.cat((atom2, atom1, globals), dim=1))
        atom2_g = self.convbn1(atom2_g)
        atom2_filter, atom2_core = atom2_g.chunk(2, dim = 1)
        atom2_filter = self.convsigmoid(atom2_filter)
        atom2_core = self.convsoftplus1(atom2_core)
        atom2_sumed = atom2_filter * atom2_core
        atom2_sumed = self.convbn2(atom2_sumed)
        atom2_g = self.convsoftplus2(atom2 + atom2_sumed)
        
        return atom1_g, atom2_g
        
class AtomPoolGNN(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, nconv = 3, outsize = 1):
        super().__init__()
        
        self.convs = nn.ModuleList([ReConvLayer(insize, arilen) for i in range(nconv)])        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(arilen, width),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width, width//funnel),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(width//funnel, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        features = torch.split(x, self.arilen, dim = 1)
        assert(len(features) > 2)
        atom1 = features[0]
        atom2 = features[1]
        globals = features[2]
        for conv_func in self.convs:
            atom1, atom2 = conv_func(atom1, atom2, globals)
        atoms_pooled = torch.mean(torch.stack((atom1,atom2)), dim = 0)
        logits = self.linear_relu_stack(atoms_pooled)
        return logits
        
    def init_weights(self, initfunct, *args, **kwargs):
        self.initfunct = initfunct
        self.apply(lambda m: self._init_weights(m, *args, **kwargs))
        
    def _init_weights(self, module, *args, **kwargs):
        if isinstance(module, nn.Linear):
            self.initfunct(module.weight, *args, **kwargs)
            if module.bias is not None:
                module.bias.data.zero_
                
class ReConvLayer2(nn.Module):
    def __init__(self, insize, arilen):
        super(ReConvLayer2,self).__init__()
        
        self.convfc = nn.Linear(insize-1, arilen*2)
        self.convsigmoid = nn.Sigmoid()
        self.convsoftplus1 = nn.Softplus()
        self.convsoftplus2 = nn.Softplus()
        self.convbn1 = nn.BatchNorm1d(arilen*2)
        self.convbn2 = nn.BatchNorm1d(arilen)
    def forward(self, atom1, atom2, nbrs, globals):
        atom1_g = self.convfc(torch.cat((atom1, atom2, globals),dim =1))
        atom1_g = self.convbn1(atom1_g)
        atom1_filter, atom1_core = atom1_g.chunk(2, dim = 1)
        atom1_filter = self.convsigmoid(atom1_filter)
        atom1_core = self.convsoftplus1(atom1_core)
        atom1_sumed = atom1_filter * atom1_core * nbrs
        atom1_sumed = self.convbn2(atom1_sumed)
        atom1_g = self.convsoftplus2(atom1 + atom1_sumed)
        
        atom2_g = self.convfc(torch.cat((atom2, atom1, globals), dim=1))
        atom2_g = self.convbn1(atom2_g)
        atom2_filter, atom2_core = atom2_g.chunk(2, dim = 1)
        atom2_filter = self.convsigmoid(atom2_filter)
        atom2_core = self.convsoftplus1(atom2_core)
        atom2_sumed = atom2_filter * atom2_core * nbrs
        atom2_sumed = self.convbn2(atom2_sumed)
        atom2_g = self.convsoftplus2(atom2 + atom2_sumed)
        
        return atom1_g, atom2_g
        
class AtomPoolGNN2(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, nconv = 3, outsize = 1):
        super().__init__()
        
        self.convs = nn.ModuleList([ReConvLayer2(insize, arilen) for i in range(nconv)])        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(arilen, width),
            nn.Dropout(p=dropout),
            nn.GELU(),
            # nn.Linear(width, width//funnel),
            # nn.Dropout(p=dropout),
            # nn.GELU(),
            nn.Linear(width, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        features = torch.split(x, self.arilen, dim = 1)
        assert(len(features) > 2)
        atom1 = features[0]
        atom2 = features[1]
        globals = features[2][:,1:]
        nbrs = features[2][:,:1]
        for conv_func in self.convs:
            atom1, atom2 = conv_func(atom1, atom2, nbrs, globals)
        atoms_pooled = torch.mean(torch.stack((atom1,atom2)), dim = 0)
        logits = self.linear_relu_stack(atoms_pooled)
        return logits
        
    def init_weights(self, initfunct, *args, **kwargs):
        self.initfunct = initfunct
        self.apply(lambda m: self._init_weights(m, *args, **kwargs))
        
    def _init_weights(self, module, *args, **kwargs):
        if isinstance(module, nn.Linear):
            self.initfunct(module.weight, *args, **kwargs)
            if module.bias is not None:
                module.bias.data.zero_
                
class LinearModel(nn.Module):
    def __init__(self, insize = 26, outsize = 1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(insize, outsize)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
        
class AtomPoolGNN_numsum(nn.Module):
    def __init__(self, insize = 26, dropout = 0.15, width = 256, funnel = 1, arilen = 10, nconv = 3, outsize = 1):
        super().__init__()
        
        self.convs = nn.ModuleList([ReConvLayer2(insize, arilen) for i in range(nconv)])        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(arilen, width),
            nn.Dropout(p=dropout),
            nn.Softplus(),
            nn.Linear(width, outsize)
        )
        self.arilen = arilen

    def forward(self, x):
        features = torch.split(x, self.arilen, dim = 1)
        assert(len(features) > 2)
        atom1 = features[0]
        atom2 = features[1]
        globals = features[2][:,1:]
        nbrs = features[2][:,:1]
        for conv_func in self.convs:
            atom1, atom2 = conv_func(atom1, atom2, nbrs, globals)
        atom1 = self.linear_relu_stack(atom1)
        atom2 = self.linear_relu_stack(atom2)
        logits = atom1 + atom2
        return logits
        
    def init_weights(self, initfunct, *args, **kwargs):
        self.initfunct = initfunct
        self.apply(lambda m: self._init_weights(m, *args, **kwargs))
        
    def _init_weights(self, module, *args, **kwargs):
        if isinstance(module, nn.Linear):
            self.initfunct(module.weight, *args, **kwargs)
            if module.bias is not None:
                module.bias.data.zero_