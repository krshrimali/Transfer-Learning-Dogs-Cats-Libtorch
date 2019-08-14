
import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
# import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time

folder_path = "/Users/krshrimali/Documents/krshrimali-blogs/dataset/train/train_python/"
transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
data = datasets.ImageFolder(root = os.path.join(folder_path), transform = transform)

batch_size = 4
data_loader = torch.utils.data.DataLoader(dataset=data, batch_size = batch_size, shuffle = True)

model = models.resnet18(pretrained = True)

for parma in model.parameters():
    parma.requires_grad = False

model.fc = torch.nn.Linear(512, 2)

for param in model.fc.parameters():
	param.requires_grad = True

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters())

n_epochs = 15

for epoch in range(n_epochs):
    mse = 0.0
    acc = 0
    batch_index = 0

    for data_batch in data_loader:
        batch_index += 1
        image, label = data_batch
        
        optimizer.zero_grad()

        output = model(image)
        _, predicted_label = torch.max(output.data, 1)
        
        loss = cost(output, label)
        
        loss.backward()
        optimizer.step()

        mse += loss.item() # data[0]
        acc += torch.sum(predicted_label == label.data)
    
    mse = mse/len(data)
    acc = 100*acc/len(data)
    
    print("Epoch: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1, n_epochs, mse, acc))
