import os
import time
from collections import deque, Counter

import cv2
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from picamera2 import Picamera2


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


def predict(frame, model, transform, class_names, device, conf_threshold=0.8):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        prob = torch.softmax(pred, dim=1)
        max_prob, idx = torch.max(prob, dim=1)

    confidence = max_prob.item()
    if confidence < conf_threshold:
        return "No Gesture", confidence

    return class_names[idx.item()], confidence


def get_center_roi(frame, box_size=250):
    h, w, _ = frame.shape

    cx = w // 2
    cy = h // 2

    half = box_size // 2
    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


if __name__ == "__main__":
    device = torch.device("cpu")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "model", "model.pth")

    class_names = ["Down", "Fist", "Left", "Right", "Up"]

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    model = ImageModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    history = deque(maxlen=8)

    # Picamera2 setup for Raspberry Pi CSI camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(1.0)  # let camera warm up

    print("Camera started. Press 'q' to exit.")

    try:
        while True:
            frame = picam2.capture_array()

            # Optional: mirror the frame for a more natural experience
            frame = cv2.flip(frame, 1)

            roi, (x1, y1, x2, y2) = get_center_roi(frame, box_size=250)

            label, confidence = predict(
                roi, model, transform, class_names, device, conf_threshold=0.8
            )

            history.append(label)
            stable_label = Counter(history).most_common(1)[0][0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"{stable_label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Gesture Recognition - Raspberry Pi", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    finally:
        cv2.destroyAllWindows()
        picam2.stop()