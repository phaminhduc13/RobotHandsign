import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import sys

# Fix Unicode for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

CLASSES = ['WAVE', 'FIST', 'POINT']

GESTURE_ACTIONS = {
    'WAVE': 'DUNG / STOP',
    'FIST': 'NAM TAY / GRIP',
    'POINT': 'RE TRAI'
}

GESTURE_COLORS = {
    'WAVE': (0, 255, 255),
    'FIST': (0, 0, 255),
    'POINT': (0, 255, 0)
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def detect_hand_region(frame):
    """Detect hand region using multiple methods"""
    h, w = frame.shape[:2]

    # Method 1: HSV skin detection (broader range)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Expanded skin color range
    lower_skin1 = np.array([0, 15, 50], dtype=np.uint8)
    upper_skin1 = np.array([30, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)

    # Method 2: YCrCb skin detection
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_skin2 = np.array((0, 135, 85), dtype=np.uint8)
    upper_skin2 = np.array((255, 180, 135), dtype=np.uint8)
    mask2 = cv2.inRange(ycrcb, lower_skin2, upper_skin2)

    # Combine masks
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, skin_mask

    # Find largest contour
    max_area = 0
    best_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area and area > 2000:  # Lowered threshold
            max_area = area
            best_contour = cnt

    if best_contour is None:
        return None, skin_mask

    # Get bounding rectangle
    x, y, cw, ch = cv2.boundingRect(best_contour)

    # Relaxed aspect ratio check
    aspect_ratio = float(cw) / ch if ch > 0 else 0

    return (x, y, cw, ch), skin_mask

def main():
    model = CNN()
    try:
        model.load_state_dict(torch.load('models/gesture_classifier.pth', weights_only=True))
        model.eval()
        print("Da load model: models/gesture_classifier.pth")
    except Exception as e:
        print(f"Loi load model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc webcam!")
        return

    print("\n=== DEBUG MODE ===")
    print("Press 'q' to quit")
    print("-" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)

            # Detect hand
            hand_region, skin_mask = detect_hand_region(frame)

            # Create debug windows
            debug_frame = frame.copy()

            if hand_region is not None:
                x, y, w, h = hand_region

                # Draw green rectangle
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # Extract and process ROI
                roi = frame[y:y+h, x:x+w]

                if roi.size > 0:
                    # Resize & normalize
                    img = cv2.resize(roi, (64, 64))
                    img = img.astype('float32') / 255.0
                    img = img.transpose(2, 0, 1)
                    img = torch.FloatTensor(img).unsqueeze(0)

                    # Predict
                    with torch.no_grad():
                        outputs = model(img)
                        probs = torch.softmax(outputs, dim=1)[0].numpy()
                        class_idx = probs.argmax()
                        label = CLASSES[class_idx]

                    # Draw info
                    cv2.putText(debug_frame, f"HAND DETECTED ({w}x{h})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(debug_frame, f"Gesture: {label} ({probs[class_idx]*100:.1f}%)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(debug_frame, "ROI EMPTY", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(debug_frame, "NO HAND DETECTED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Resize for display
            display = cv2.resize(debug_frame, (640, 480))

            # Show skin mask in separate window
            mask_display = cv2.resize(skin_mask, (320, 240))
            mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Skin Mask", mask_colored)

            # Show main window
            cv2.imshow("Hand Detection Debug", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nDa thoat!")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera da dong.")

if __name__ == "__main__":
    main()