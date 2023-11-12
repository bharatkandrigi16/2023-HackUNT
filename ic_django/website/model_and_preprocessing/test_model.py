import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn
from .train_model import classes, test_loader, criterion, os
from PIL import Image

print('CWD:',os.getcwd())
model_path = 'website/model_and_preprocessing/model/skin_disease_classification_model.pth'
model = models.densenet201(pretrained=False)
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, len(classes))
model.load_state_dict(torch.load(os.path.join(os.getcwd(), model_path)))
print(model)

test_loss = 0.0
class_correct = np.zeros((len(classes)))
class_total = np.zeros((len(classes)))
model.eval()

for data, target in test_loader:
    if torch.cuda.is_available():
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
    if len(target) == 12:
        for i in range(12):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
test_loss /= len(test_loader.dataset)
print("Test Loss: {:.6f}\n".format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print("Test Accuracy of {}: {} ({}/{})".format(
            classes[i], 100*class_correct[i]/class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])
        ))
    else:
        print("Test Accuracy of {}: N/A (since there are no examples)".format(
            classes[i]
        ))
        print("Test Accuracy overall: {} ({}/{})".format(
            classes[i], 100*class_correct[i]/class_total[i], 
            np.sum(class_correct[i]), np.sum(class_total[i])
        ))

def process_single_image(path):
    image = Image.open(path)
    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    #Apply the transformations
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    #Inference
    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()
        output = model(input_batch)
    #Post-processing
    _, predicted_idx = torch.max(output, 1)
    predicted_class = classes[predicted_idx.item()]
    return predicted_class

