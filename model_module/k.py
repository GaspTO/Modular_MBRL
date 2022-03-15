from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import torch

def cnn(
    input_shape:tuple,
    num_channels:list,
    kernel_sizes:List[Tuple],
    mlp_layer_sizes:list,
    device,
    activation=torch.nn.ReLU,
    output_activation=torch.nn.Identity,
):
    assert len(num_channels) == len(kernel_sizes)
    assert len(input_shape) == 3
    dims = [input_shape[1],input_shape[2]]
    for i in range(len(kernel_sizes)):
        if isinstance(kernel_sizes[i],int):
            kernel_sizes[i] = (kernel_sizes[i],kernel_sizes[i])
        dims[0] = dims[0] - (kernel_sizes[i][0]-1)
        dims[1] = dims[1] - (kernel_sizes[i][1]-1)
        if dims[0] <= 0 or dims[1] <= 0:
            raise ValueError("Input Shape is not big enough")

    stride = 1 #assuming stride 1 for now
    previous_channels = input_shape[0]
    layers = []
    number_of_layers = len(num_channels) + len(mlp_layer_sizes)
    for layer_num in range(number_of_layers):
        #Layers
        if layer_num < len(num_channels):
            layers += [nn.Conv2d(previous_channels, num_channels[layer_num], kernel_sizes[layer_num], stride=stride)]
            previous_channels = num_channels[layer_num]
        else:
            mlp_layer_num = layer_num-len(num_channels)
            layers += [torch.nn.Linear(mlp_layer_sizes[mlp_layer_num], mlp_layer_sizes[mlp_layer_num + 1]).to(device)]
        #Activations
        if layer_num < len(num_channels) + len(mlp_layer_sizes) - 1:
            if layer_num < len(num_channels):
                layers += [torch.nn.BatchNorm2d(num_channels[layer_num]),activation()]
            else:
                mlp_layer_num = layer_num-len(num_channels)
                layers += [torch.nn.BatchNorm1d(mlp_layer_sizes[mlp_layer_num]),activation()]
        else:
            layers += [output_activation()]
    return torch.nn.Sequential(*layers)


model = cnn((1,10,10),[100,300,500],[3,3,3],[],torch.device("cpu"))

print(model)


x = torch.randn(1,1,7,7)

print(model(x).shape)