"""
This python script converts the network into Script Module
"""

import torch
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
for param in resnet18.parameters():
	param.requires_grad = False
resnet18.fc = torch.nn.Linear(512, 2)
for param in resnet18.fc.parameters():
	param.requires_grad = True

example_input = torch.rand(1, 3, 224, 224)
script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet18_with_last_layer.pt')

# print(list(resnet18.children()))
# resnet18 = torch.nn.Sequential(*list(resnet18.children())) # Take all layers except the last one
# list(resnet18.children())[-1] = torch.nn.Linear(512, 2)
# print(list(resnet18.children())[-1])
'''
example_input = torch.rand(1, 3, 224, 224)

script_module = torch.jit.trace(resnet18, example_input)
script_module.save('resnet18_without_lastlayer.pt')
'''
