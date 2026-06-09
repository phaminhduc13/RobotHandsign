import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from collections import Counter



GESTURE_ACTIONS = {
    'WAVE': 'DUNG / STOP',
    'FIST': 'NAM TAY / GRIP',
    'POINT_LEFT': 'RE TRAI',
    'POINT_RIGHT': 'RE PHAI'
}

GESTURE_COLORS = {
    'WAVE': (0, 255, 255),
    'FIST': (0, 0, 255),
    'POINT_LEFT': (0, 255, 0),
    'POINT_RIGHT': (255, 0, 0)
}

CLASSES = ['WAVE', 'FIST', 'POINT']

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

def count_fingers_from_landmarks(landmarks, handedness):
    """Count extended fingers using distance-based detection."""
    tips = [4, 8, 12, 16, 20]  # All finger tips
    mcps = [2, 5, 9, 13, 17]  # MCP positions

    finger_count = 0

    for i in range(5):
        mcp = landmarks[mcps[i]]
        tip = landmarks[tips[i]]
        dx = tip.x - mcp.x
        dy = tip.y - mcp.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance > 0.12:
            finger_count += 1

    return finger_count

def is_pointing_gesture(landmarks):
    """Check if hand is making a pointing gesture (only 1 finger extended)."""
    tips = [4, 8, 12, 16, 20]  # All finger tips
    mcps = [2, 5, 9, 13, 17]  # MCP positions

    extended_count = 0
    extended_index = -1

    for i in range(5):
        mcp = landmarks[mcps[i]]
        tip = landmarks[tips[i]]
        dx = tip.x - mcp.x
        dy = tip.y - mcp.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance > 0.12:
            extended_count += 1
            extended_index = i

    # Pointing = exactly 1 finger extended (index = 0)
    return extended_count == 1 and extended_index == 1

def get_pointing_direction(landmarks, handedness):
    """Get pointing direction: LEFT, RIGHT, or UP.
    Account for frame flip - compare index_tip.x to wrist.x."""
    wrist = landmarks[0]
    index_tip = landmarks[8]

    dy = index_tip.y - wrist.y

    # UP: index significantly above wrist
    if dy < -0.12:
        return 'UP'

    # Frame is flipped, so direct comparison works
    # LEFT hand on left side of frame (after flip) -> pointing left
    # RIGHT hand on right side of frame (after flip) -> pointing right
    if index_tip.x < wrist.x:
        return 'LEFT'
    else:
        return 'RIGHT'

def detect_gesture_from_landmarks(hand_landmarks, handedness):
    """Determine gesture from hand landmarks."""
    landmarks = hand_landmarks

    tips = [4, 8, 12, 16, 20]
    mcps = [2, 5, 9, 13, 17]

    extended_count = 0
    extended_index = -1
    index_distance = 0

    for i in range(5):
        mcp = landmarks[mcps[i]]
        tip = landmarks[tips[i]]
        dx = tip.x - mcp.x
        dy = tip.y - mcp.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance > 0.10:
            extended_count += 1
            if i == 1:  # Index finger
                index_distance = distance

    # Pointing = index is clearly extended (distance > 0.10) and is the longest
    if extended_count >= 1:
        # Check if index is extended and clearly longer than others
        if index_distance > 0.10:
            direction = get_pointing_direction(landmarks, handedness)
            if direction == 'LEFT':
                return 'POINT_LEFT'
            elif direction == 'RIGHT':
                return 'POINT_RIGHT'
            else:
                if extended_count == 1:
                    return 'FIST'
                else:
                    return 'WAVE'

    # Not pointing - check finger count for wave/fist
    if extended_count >= 4:
        return 'WAVE'
    elif extended_count <= 1:
        return 'FIST'
    else:
        return 'WAVE'

def get_hand_bounding_box(landmarks, frame_shape):
    """Get bounding box from hand landmarks."""
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = int(min(xs)) - 20, int(max(xs)) + 20
    y_min, y_max = int(min(ys)) - 20, int(max(ys)) + 20
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    return x_min, y_min, x_max - x_min, y_max - y_min

def print_result(label, probs):
    bar_char = '#'
    print(f"\n{'='*50}")
    print(f"  KET QUA: {label}")
    print(f"{'='*50}")
    for cls, prob in zip(CLASSES, probs):
        bar = bar_char * int(prob * 30)
        print(f"  {cls:10s}: {prob*100:5.1f}% [{bar}]")

def draw_gui(frame, label, probs, action, hand_detected, finger_count):
    display = cv2.resize(frame, (640, 480))

    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (630, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    color = GESTURE_COLORS.get(label, (255, 255, 255))

    if hand_detected:
        cv2.putText(display, "HAND DETECTED", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(display, "MOVE YOUR HAND", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(display, f"Gesture: {label}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(display, f"Action: {action}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(display, f"Fingers: {finger_count}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    max_prob = max(probs)
    bar_width = int(max_prob * 300)
    cv2.rectangle(display, (20, 155), (320, 165), (50, 50, 50), -1)
    cv2.rectangle(display, (20, 155), (20 + bar_width, 165), color, -1)
    cv2.putText(display, f"{max_prob*100:.1f}%", (330, 163),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    y_pos = 460
    for cls, prob in zip(CLASSES, probs):
        bar = '#' * int(prob * 20)
        text = f"{cls}: {prob*100:5.1f}% [{bar}]"
        cv2.putText(display, text, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_pos -= 18

    return display

def main():
    model = CNN()
    try:
        model.load_state_dict(torch.load('models/gesture_classifier.pth', weights_only=True))
        model.eval()
        print("Da load model: models/gesture_classifier.pth")
    except Exception as e:
        print(f"Loi load model: {e}")
        return

    # MediaPipe Hands - using new tasks API
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)
    print("Da khoi tao MediaPipe HandLandmarker")

    # Open camera
    cap = None
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        for idx in [0, 1]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    time.sleep(0.5)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"Mo webcam thanh cong: index={idx}, backend={backend}")
                        break
                    else:
                        cap.release()
                        cap = None
                else:
                    cap.release()
                    cap = None
            except Exception:
                cap = None
        if cap is not None:
            break

    if cap is None or not cap.isOpened():
        print("Khong mo duoc webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=== Demo Real-time Hand Gesture Recognition ===")
    print("Dang chay... nhan 'q' de thoat")
    print(f"Classes: {CLASSES}")
    print("-" * 50)

    last_print = 0
    interval = 0.5

    # Gesture smoothing - require more consecutive frames to confirm
    confirm_threshold = 8
    gesture_history = []
    confirmed_gesture = 'WAVE'  # Default to WAVE when no hand detected

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)

            if results.hand_landmarks and results.handedness:
                hand_landmarks = results.hand_landmarks[0]
                handedness_label = results.handedness[0][0].category_name

                # Draw landmarks
                drawing_utils.draw_landmarks(frame, hand_landmarks)

                # Get bounding box
                x, y, w, h = get_hand_bounding_box(hand_landmarks, frame.shape)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Detect gesture from landmarks
                finger_count = count_fingers_from_landmarks(hand_landmarks, handedness_label)
                finger_gesture = detect_gesture_from_landmarks(hand_landmarks, handedness_label)

                # Smoothing
                gesture_history.append(finger_gesture)
                if len(gesture_history) > 10:
                    gesture_history.pop(0)

                gesture_counts = Counter(gesture_history)
                most_common = gesture_counts.most_common(1)[0][0]
                if gesture_counts[most_common] >= confirm_threshold:
                    confirmed_gesture = most_common

                label = confirmed_gesture

                # CNN prediction for probability display
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    img = cv2.resize(roi, (64, 64))
                    img = img.astype('float32') / 255.0
                    img = img.transpose(2, 0, 1)
                    img = torch.FloatTensor(img).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(img)
                        probs = torch.softmax(outputs, dim=1)[0].numpy()
                else:
                    probs = [0.33, 0.33, 0.34]

                hand_detected = True
            else:
                probs = [0.33, 0.33, 0.34]
                label = confirmed_gesture
                finger_count = -1
                hand_detected = False

            action = GESTURE_ACTIONS.get(label, 'UNKNOWN')

            display = draw_gui(frame, label, probs, action, hand_detected, finger_count)
            cv2.imshow('Hand Gesture Recognition', display)

            current_time = time.time()
            if current_time - last_print >= interval and hand_detected:
                print_result(label, probs)
                print(f"  ACTION: {action}")
                print(f"  FINGERS: {finger_count}")
                last_print = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nDa thoat!")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Camera da dong.")

if __name__ == "__main__":
    main()
