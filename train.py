#!/usr/bin/python

import numpy as np
import torch, os
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

device = torch.device('cpu')
model = models.resnet50(pretrained=True)

data_dir = '/chess'

def load_split_train_test(datadir, valid_size = .2):
    copy_and_rotate_images(datadir)
    train_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx = indices[split:]
    test_idx = indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=64)
    return trainloader, testloader

'''
Creates copies of every .jpg image in the dataset rotated by 90, 180 and 270 degrees.
A suffix is added to the end of the new files' name.
'''
def copy_and_rotate_images(datadir):
    CATEGORIES = ["black_pawns", "black_knights", "black_rooks", "black_bishops", "black_queens", "black_kings", "white_pawns", "white_knights", "white_rooks", "white_bishops", "white_queens", "white_kings", "empty_tiles"]
    for cat in range(len(CATEGORIES)):
        thisdir = datadir + "/" + CATEGORIES[cat]
        for file in os.listdir(thisdir):
            if (os.fsdecode(file).endswith(".jpg")):
                PATH = thisdir + "/" + file
                CUT_PATH = PATH[:-4] # Image path with the ".jpg" suffix emitted

                # Skip if this image has already been rotated before or this image is a rotated clone
                if os.path.exists(CUT_PATH + "_90.jpg") or CUT_PATH.endswith(("_90", "_180", "_270")):
                    continue

                img = Image.open(PATH)

                img_90 = img.transpose(Image.ROTATE_90)
                img_180 = img.transpose(Image.ROTATE_180)
                img_270 = img.transpose(Image.ROTATE_270)

                img_90.save(CUT_PATH + '_90.jpg')
                img_180.save(CUT_PATH + '_180.jpg')
                img_270.save(CUT_PATH + '_270.jpg')

trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)

device = torch.device('cpu')
model = models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 13), # Used to be "10),"
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses = []
test_losses = []

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'pytorch_chessmodel.pth')

