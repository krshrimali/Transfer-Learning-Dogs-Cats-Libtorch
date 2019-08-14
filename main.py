
import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
# import matplotlib.pyplot as plt
from torch.autograd import Variable 
import time


path = "/Users/krshrimali/Documents/krshrimali-blogs/dataset/train/"
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

data_image = {x:datasets.ImageFolder(root = os.path.join(path,x),
                                     transform = transform)
              for x in ["train_python"]}

data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],
                                                batch_size = 4,
                                                shuffle = True)
                     for x in ["train_python"]}


classes = data_image["train_python"].classes
classes_index = data_image["train_python"].class_to_idx
print(classes)
print(classes_index)

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
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-"*10)
    for param in ["train_python"]:
        if param == "train_python":
            model.train = True
        else:
            model.train = False
            
        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader_image[param]:
            batch += 1
            X,y = data
            X,y = Variable(X), Variable(y)
            
            optimizer.zero_grad()
            y_pred = model(X)
            _,pred = torch.max(y_pred.data, 1)
            
            loss = cost(y_pred,y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item() # data[0]
            running_correct += torch.sum(pred == y.data)
            if batch%5 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                batch, running_loss/(4*batch), 100*running_correct/(4*batch)))
                
        epoch_loss = running_loss/len(data_image[param])
        epoch_correct = 100*running_correct/len(data_image[param])
        
        print("{} Loss:{:.4f}, Correct:{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time//60, now_time%60))
