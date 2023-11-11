import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import datasets, transforms, models
import numpy as np
from ignite.engine import Engine, Events, create_supervised_trainer
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Loss, RunningAverage
import matplotlib.pyplot as plt

#Define path to dataset and data set classes
def get_data_loaders(dir, batch_size):
    transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor()])
    all_images= datasets.ImageFolder(dir, transform=transform)
    train_images_len = int(len(all_images)*.75)
    valid_images_len = int((len(all_images)-train_images_len)/2)
    test_images_len = int((len(all_images)-train_images_len-valid_images_len))
    train_data, val_data, test_data = random_split(all_images, [train_images_len, valid_images_len, test_images_len])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return (train_loader, val_loader, test_loader), all_images.classes


#Retrieve loader to load dataset
(train_loader, val_loader, test_loader), classes = get_data_loaders('/dataset/')
data_iter = iter(train_loader)
images, labels = data_iter.next()
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1,2,0)))
    ax.set_title(classes[labels[idx]])

model = models.densenet201(pretrained=True)
print(model)
