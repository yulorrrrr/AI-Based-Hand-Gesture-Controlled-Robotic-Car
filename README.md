# 🤖 AI-Based Hand Gesture Controlled Robotic Car
 
A deep learning project that recognizes hand gestures in real-time using a Convolutional Neural Network (CNN) and translates them into movement commands for a robotic car via Raspberry Pi and Arduino.
 

## 📋 Table of Contents
 
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Gestures](#gestures)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Hardware (Coming Soon)](#hardware-coming-soon)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Roadmap](#roadmap)
 


## Project Overview
 
This project uses a custom-trained CNN to classify 5 hand gestures captured via camera. The recognized gesture is then sent as a command to a Raspberry Pi, which communicates with an Arduino to control the motors of a small robotic car.
 
**Current Status:** ✅ Model training complete — Hardware integration in progress
 
 
## System Architecture
 
```
Camera Input
     │
     ▼
┌──────────────────────────┐
│   CNN Model              │  ← PyTorch (trained on custom dataset)
│   Image Classification   │
└──────────────────────────┘
     │
     ▼ Gesture Label
┌──────────────────────────┐
│      Raspberry Pi        │  ← Runs inference, sends serial commands
└──────────────────────────┘
     │ Serial (USB)
     ▼
┌──────────────────────────┐
│         Arduino          │  ← Controls motor driver
└──────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│      Robotic Car         │  ← Moves based on gesture
└──────────────────────────┘
```
 
---
 
## Gestures
 
| Gesture | Label | Car Action | Number |
|---------|-------|------------|--------|
| 🫳 Down  | `Down`  | Backward   | 0 |
| ✊ Fist  | `Fist`  | Stop       | 1 |
| 🫲 Left  | `Left`  | Turn Left  | 2 |
| 🫱 Right | `Right` | Turn Right | 3 |
| ✋ Up    | `Up`    | Forward    | 4 |

## Project Structure
 
```
AI-Based Hand Gesture Controlled Robotic Car/
│
├── Dataset/
│   ├── Train/
│   │   ├── Down/        # 500 images each
│   │   ├── Fist/
│   │   ├── Left/
│   │   ├── Right/
│   │   └── Up/
│   └── Test/
│       ├── Down/
│       ├── Fist/
│       ├── Left/
│       ├── Right/
│       └── Up/
│
├── model/
│   └── model.pth        # Saved trained model
│
├── Image_Classification.py   # Model definition, training, evaluation
└── README.md
```
 
---
 
## Dataset
 
- **Total training images:** 2,500 (500 per class)
- **Classes:** Down, Fist, Left, Right, Up
- **Image size:** Resized to 128×128 RGB
- **Source:** Custom collected — photos taken by the team
 
### Data Augmentation (applied during training)
- Random rotation (±15°)
- Color jitter (brightness & contrast)
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
 
---
 
## Model
 
A custom CNN built with PyTorch.
 
### Architecture
 
```
Input (3, 128, 128)
  → Conv2d(3→16) + BatchNorm + ReLU + MaxPool   →  (16, 64, 64)
  → Conv2d(16→32) + BatchNorm + ReLU + MaxPool  →  (32, 32, 32)
  → Conv2d(32→64) + BatchNorm + ReLU + MaxPool  →  (64, 16, 16)
  → Flatten  →  16384
  → Linear(16384→256) + ReLU + Dropout(0.5)
  → Linear(256→64) + ReLU
  → Linear(64→5)
  → Output (5 classes)
```
 
### Training Config
 
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Weight Decay | 1e-4 |
| Batch Size | 8 |
| Epochs | 20 |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau |
 
---
 
## Hardware (Coming Soon)
 
The following hardware integration is planned for the next phase:
 
### Components
- **Raspberry Pi 5 8GB** - runs the trained CNN model and camera input
- **Arduino Uno** - receives serial commands, controls motor driver
- **Pi Camera OV5647** - real-time hand gesture capture
- **HC - 06 Bluetoo Module** - connect raspberry pi and arduino 
- **4 MG90S Motor Driver** - drives 4 DC motors
- **Robotic Car Chassis** - 4WD platform
- **Power Bank / Battery Pack** - powers Raspberry Pi
- **9V Battery** - powers Arduino + motors
 
### Communication Flow
- Raspberry Pi captures frame → runs inference → sends command string over serial (e.g., `"FORWARD"`)
- Arduino reads serial → maps command to motor directions
 
---
 
## Installation
 
### Requirements
 
```bash
Python 3.10+
PyTorch
torchvision
matplotlib
torchsummary
```
 
### Setup
 
```bash
# Clone the repository
git clone https://github.com/yourusername/hand-gesture-car.git
cd hand-gesture-car
 
# Install dependencies
pip install torch torchvision matplotlib torchsummary
```
 
---
 
## Usage
 
### Train the model
 
In `Image_Classification.py`, uncomment `train(train_data)` and comment out `evaluate(test_data)`:
 
```python
if __name__ == '__main__':
    train_data, test_data = create_dataset()
    train(train_data)       # ← uncomment to train
    # evaluate(test_data)   # ← comment out
```
 
Then run:
```bash
python Image_Classification.py
```
 
### Evaluate the model
 
```python
if __name__ == '__main__':
    train_data, test_data = create_dataset()
    # train(train_data)     # ← comment out
    evaluate(test_data)     # ← uncomment to evaluate
```
 
---
 
## Results
 
| Epoch | Loss   | Train Accuracy |
|-------|--------|----------------|
| 1     | 0.7856 | 66.72%         |
| 5     | 0.0126 | 99.60%         |
| 10    | 0.0082 | 99.72%         |
 
> ⚠️ Note: High training accuracy with low test accuracy indicates overfitting. Data augmentation and BatchNorm have been added to address this — updated results coming soon.
 
---
 
## Roadmap
 
- [x] Collect custom hand gesture dataset
- [x] Build and train CNN model
- [x] Add data augmentation & BatchNorm to reduce overfitting
- [ ] Evaluate improved model performance
- [ ] Set up Raspberry Pi with camera for real-time inference
- [ ] Implement serial communication between Raspberry Pi and Arduino
- [ ] Write Arduino motor control code
- [ ] Assemble robotic car hardware
- [ ] End-to-end integration test
- [ ] Demo video
 
---
 
## Author
 
**Christine** — University of Waterloo, Mechatronics Engineering
Project: AI-Based Hand Gesture Controlled Robotic Car
 
