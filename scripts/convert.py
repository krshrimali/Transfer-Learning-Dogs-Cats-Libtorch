"""
This python script converts the network into Script Module
"""

import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)

print(list(resnet18.children())[-1])
resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1]) # Take all layers except the last one
example_input = torch.rand(1, 3, 224, 224)

script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet18_without_lastlayer.pt')
