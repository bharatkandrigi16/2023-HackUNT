from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import os

print(os.getcwd())
print(os.listdir())

#Define path to dataset and data set classes
def get_data_loaders(dir, batch_size):
    transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor()])
    absolute_dir = os.path.join(os.getcwd(), dir)  # Constructing an absolute path
    all_images= datasets.ImageFolder(absolute_dir, transform=transform)
    train_images_len = int(len(all_images)*.75)
    valid_images_len = int((len(all_images)-train_images_len)/2)
    test_images_len = int((len(all_images)-train_images_len-valid_images_len))
    train_data, val_data, test_data = random_split(all_images, [train_images_len, valid_images_len, test_images_len])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return (train_loader, val_loader, test_loader), all_images.classes


#Retrieve loader to load dataset
(train_loader, val_loader, test_loader), classes = get_data_loaders('website/model_and_preprocessing/dataset', 20)
data_iter = iter(train_loader)
images, labels = next(data_iter)
fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1,2,0)))
    ax.set_title(classes[labels[idx]])


#Training Code:

import torch.nn as nn
import torch.optim as optim
import torch 

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Loss, ConfusionMatrix


model = models.densenet201(pretrained=True)
#print(model)
# (classifier): Linear(in_features=1920, out_features=1000, bias=True) - need to change
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 10

training_history = {'accuracy': [], 'loss': []}
validation_history = {'accuracy': [], 'loss': []}
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model, device=device, metrics={'accuracy': Accuracy(), 'loss': Loss(criterion), 'cm': ConfusionMatrix(len(classes))})

# @trainer.on(Events.ITERATION_COMPLETED)
# def log_a_dot(engine):
#   if engine.state.iteration % 100 == 0:
#     print(".",end="")

# @trainer.on(Events.EPOCH_COMPLETED)
# def log_training_results(trainer):
#     evaluator.run(train_loader)
#     metrics = evaluator.state.metrics
#     accuracy = metrics['accuracy']*100
#     loss = metrics['loss']
#     training_history['accuracy'].append(accuracy)
#     training_history['loss'].append(loss)
#     print(f'Training Results - Epoch: {trainer.state.epoch} Avg accuracy: {accuracy}\nLoss: {loss}')

# @trainer.on(Events.EPOCH_COMPLETED)
# def log_validation_results(trainer):
#     evaluator.run(val_loader)
#     metrics = evaluator.state.metrics
#     accuracy = metrics['accuracy']*100
#     loss = metrics['loss']
#     validation_history['accuracy'].append(accuracy)
#     validation_history['loss'].append(loss)
#     print()
#     print(f'Validation Results - Epoch: {trainer.state.epoch} Avg accuracy: {accuracy}\nLoss: {loss}')

# trainer.run(train_loader, max_epochs=6)

#Plot training data batch performance as a function for each succession in training(epoch) and monitor improvement through each epoch
# fig, axs = plt.subplots(2,2)
# fig.set_figheight(6)
# fig.set_figwidth(14)
# axs[0, 0].plot(training_history['accuracy'])
# axs[0, 0].set_title("Training Accuracy")
# axs[0, 1].plot(training_history['accuracy'])
# axs[0, 1].set_title("Validation Accuracy")
# axs[1, 0].plot(training_history['loss'])
# axs[1, 0].set_title("Training Loss")
# axs[1, 1].plot(training_history['loss'])
# axs[1, 1].set_title("Validation Loss")

# torch.save(model.state_dict(), 'skin_disease_classification_model.pth')
