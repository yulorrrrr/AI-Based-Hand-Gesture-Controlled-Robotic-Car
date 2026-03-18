#import
import os
import test
import torch
from torch import nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

BATCH_SIZE = 8

#1. create dataset
def create_dataset():
    # Get the absolute path of the folder that contains this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    #transform the size of the image to (3,128,128)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_data = ImageFolder(root=os.path.join(base_dir, 'Dataset', 'Train'), transform=train_transform)
    test_data  = ImageFolder(root=os.path.join(base_dir, 'Dataset', 'Test'),  transform=test_transform)

    return train_data, test_data

#2. build convolutional neural network
class ImageModel(nn.Module):
    def __init__(self):
        #Initialize parent class
        super().__init__()

        #prevent overfitting
        self.dropout = nn.Dropout(0.5)

        #Convolutional & pooling layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) #（16， 128， 128）
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # (16, 64, 64)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # (32, 64, 64)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) #（32， 32， 32）

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) #（64，32， 32）
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2) # （64，16，16） 64*16*16 = 16384
        
        #Fully connected layer
        self.linear1 = nn.Linear(16384, 256)
        self.linear2 = nn.Linear(256, 64)
        self.output = nn.Linear(64,5)

    def forward(self, x):
        #convolutional + activation function + pooling layer
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.reshape(x.shape[0], -1) #(batch_size, 64, 16, 16) -> (batch_size, 16384)

        #fully connect layer (only deal with 2d data)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))

        return self.output(x)

#3. Model Training
def train(train_data):
    dataLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    criterion = nn.CrossEntropyLoss() #cross entropy loss = softmax() + loss computation
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    epochs = 10
    
    #training loop
    model.train() #set the model to training mode
    for epoch in range(epochs):
        total_lost, total_sample, total_correct, start = 0.0, 0, 0, 0
        start =time.time()
        for x, y in dataLoader:
            y_pred = model(x)
            loss = criterion(y_pred, y)

            #backpropagation
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            total_correct += (y_pred.argmax(1) == y).sum()
            total_lost += loss.item() * len(y)
            total_sample += len(y)
        
        print(f'epoch: {epoch+1}, loss: {total_lost/total_sample:.4f}, accuracy: {total_correct/total_sample:.4f}, time: {(time.time()-start):.2f} seconds')
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(model.state_dict(), os.path.join(base_dir, 'model', 'model.pth'))

#4. Model Testing
#def evaluate(test_data):
    dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model.load_state_dict(torch.load(os.path.join(base_dir, 'model', 'model.pth'), weights_only=True))
    total_correct, total_sample = 0, 0

    #Testing loop
    model.eval()
    with torch.no_grad(): 
        for x, y in dataloader:
            y_pred = model(x)
            #argmax: return the index of the maximum value in the specified dimension, dim=-1 means the last dimension
            y_pred = torch.argmax(y_pred, dim=-1)
            total_correct += (y_pred == y).sum()
            total_sample += len(y)

    print(f'test accuracy: {total_correct/total_sample:.4f}')


#5. Single image testing
#def predict_single(image_path):
    model = ImageModel()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model.load_state_dict(torch.load(os.path.join(base_dir, 'model', 'model.pth'), weights_only=True))
    model.eval()

    class_names = ['Down', 'Fist', 'Left', 'Right', 'Up']

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # 加 batch 维度

    with torch.no_grad():
        output = model(img_tensor)
        confidence = torch.softmax(output, dim=-1)
        pred = torch.argmax(output, dim=-1).item()

    print(f'Prediction : {class_names[pred]}')
    print(f'Confidence : {confidence[0][pred]:.2%}')
    print()
    print('All classes:')
    for i, name in enumerate(class_names):
        bar = '█' * int(confidence[0][i] * 20)
        print(f'  {name:<6} {confidence[0][i]:.2%}  {bar}')

def evaluate(test_data):
    dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model.load_state_dict(torch.load(os.path.join(base_dir, 'model', 'model.pth'), weights_only=True))

    class_names = ['Down', 'Fist', 'Left', 'Right', 'Up']
    correct_per_class = [0] * 5
    total_per_class   = [0] * 5
    
    confusion = [[0]*5 for _ in range(5)]

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = torch.argmax(model(x), dim=-1)

            for true, pred in zip(y, y_pred):
                confusion[true][pred] += 1
                total_per_class[true] += 1
                if true == pred:
                    correct_per_class[true] += 1

    print(f'\n{"Class":<8} {"Correct":>8} {"Total":>8} {"Accuracy":>10}')
    print("-" * 38)

    for i, name in enumerate(class_names):
        acc = correct_per_class[i] / total_per_class[i]
        print(f'{name:<8} {correct_per_class[i]:>8} {total_per_class[i]:>8} {acc:>10.2%}')

    total_correct = sum(correct_per_class)
    total_sample  = sum(total_per_class)

    print(f'\nOverall accuracy: {total_correct/total_sample:.4f}')
    print(f'\nConfusion Matrix (row=actual, col=predicted):')

    print(f'{"":>8}', end='')
    for name in class_names:
        print(f'{name:>8}', end='')
    print()
    
    for i, name in enumerate(class_names):
        print(f'{name:>8}', end='')
        for j in range(5):
            print(f'{confusion[i][j]:>8}', end='')
        print()


#test
if __name__ == '__main__':
    train_data, test_data = create_dataset()
    print(f'train_data: {len(train_data)}')
    print(f'type of dataset: {train_data.class_to_idx}')
    
    #2. create neural network model object
    model = ImageModel()

    #check model parameters
    #summary(model, input_size=(3, 128, 128), batch_size=BATCH_SIZE)

    #3. model training
    #train(train_data)

    #4. model testing
    evaluate(test_data)

