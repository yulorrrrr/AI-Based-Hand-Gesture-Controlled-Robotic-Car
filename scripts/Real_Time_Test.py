'''
Description:
This script implements real-time hand gesture recognition using a pre-trained CNN model. It captures 
video from the webcam, processes each frame to detect hand gestures, and displays the predicted 
gesture on the screen. The model is designed to recognize five gestures: Down, Fist, Left, Right, and
Up. The script also includes a smoothing mechanism to improve prediction stability by maintaining a 
history of recent predictions and using majority voting to determine the final output. Pressing the 
ESC key will exit the program.

You can run this program to test the real-time performance of the trained model with your webcam 
before deploying it on a Raspberry Pi. 

**tips: make sure you have the model.pth file in the correct path

'''




import cv2
import os
import torch
from torchvision import transforms
from torch import nn as nn
from PIL import Image
from collections import deque, Counter

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.linear1 = nn.Linear(16384, 256)
        self.linear2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.reshape(x.shape[0], -1)

        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))

        return self.output(x)
    
def predict(frame, model, transform, class_name):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #BGR -> 
    img = Image.fromarray(img) #numpy array -> PIL image
    x = transform(img).unsqueeze(0) # (3, 128, 128) -> (1, 3, 128, 128)

    with torch.no_grad():
        pred = model(x)
        prob = torch.softmax(pred, dim=1)
        max_prob, idx = torch.max(prob, dim=1)

    if max_prob.item() < 0.8:
        return "No Gesture"
    
    return class_name[idx]
    
if __name__ == "__main__":
    model = ImageModel()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model.load_state_dict(torch.load(os.path.join(base_dir, '..','model', 'model.pth'), map_location='cpu'))
    model.eval()
   
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    class_name = ['Down', 'Fist', 'Left', 'Right', 'Up']
    history = deque(maxlen=10)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print('Camera started. Press ESC to exit.')
    
    while True:
        can_read, frame = cap.read()
        if not can_read:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        roi = frame[235:485, 515:765] 

        label = predict(roi, model, transform, class_name)
        history.append(label)
        stable_label = Counter(history).most_common(1)[0][0]    

        cv2.rectangle(frame, (515, 235), (765, 485), (255, 0, 0), 2)
        cv2.putText(frame, stable_label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) == 27: #ESC key to break
            print('Exiting...')
            break

    cap.release()
    cv2.destroyAllWindows()