import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import sys
from collections import Counter

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

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

def count_fingers(roi):
    """Count number of fingers using contour analysis"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0

    # Find largest contour (hand)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 1000:
        return 0

    # Convex hull and defects
    hull = cv2.convexHull(cnt, returnPoints=False)

    if len(hull) < 3:
        return 0

    try:
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            # Count significant defects (between fingers)
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d[0] > 10000:  # Significant depth
                    finger_count += 1
            return finger_count
    except cv2.error:
        pass

    return 0

def detect_hand_by_motion(frame, prev_frame):
    """Detect hand using motion + skin color"""
    if prev_frame is None:
        return None, None

    h, w = frame.shape[:2]
    margin = int(w * 0.20)  # 20% margin to exclude face
    center_roi = frame[margin:h-margin, margin:w-margin]

    hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)

    prev_gray = cv2.cvtColor(cv2.flip(prev_frame[margin:h-margin, margin:w-margin], 1), cv2.COLOR_BGR2GRAY)

    # Motion
    diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Skin in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.cvtColor(cv2.inRange(hsv, lower_skin, upper_skin), cv2.COLOR_GRAY2BGR)

    # Combine
    hand_mask = cv2.bitwise_and(motion_mask, cv2.cvtColor(skin_mask, cv2.COLOR_BGR2GRAY))

    kernel = np.ones((5, 5), np.uint8)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, hand_mask

    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)

        # Exclude face-like regions (square-ish, in center)
        aspect = h_cnt / w_cnt if w_cnt > 0 else 0
        is_face_like = (aspect > 0.8 and aspect < 1.4 and
                        x_cnt > 50 and x_cnt < w - 100)

        if area > max_area and area > 3000 and not is_face_like:
            max_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None, hand_mask

    x, y, cw, ch = cv2.boundingRect(best_cnt)
    return (x + margin, y + margin, cw, ch), hand_mask

def determine_gesture_by_fingers(roi):
    """Determine gesture based on finger count"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Skin color filtering
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 15, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply mask
    masked = cv2.bitwise_and(gray, gray, mask=skin_mask)
    _, thresh = cv2.threshold(masked, 50, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 'POINT'

    # Find hand contour (largest)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 2000:
        return 'POINT'

    # Get convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)

    if len(hull) < 3:
        return 'POINT'

    try:
        defects = cv2.convexityDefects(cnt, hull)
        if defects is not None:
            # Count fingers based on defects
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 10000:
                    finger_count += 1
        else:
            finger_count = 0
    except:
        finger_count = 0

    # Aspect ratio of hand region - key for distinguishing fist vs point
    h_roi, w_roi = roi.shape[:2]
    aspect = w_roi / h_roi if h_roi > 0 else 1

    # Determine gesture
    if finger_count >= 4:
        return 'WAVE'  # Open hand (lots of fingers showing)
    elif finger_count == 3:
        return 'WAVE'  # Also open hand
    elif finger_count == 2:
        # Check direction of pointing (left or right)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            # Left side of ROI = pointing left, right side = pointing right
            if cx < w_roi * 0.4:
                return 'POINT_LEFT'
            elif cx > w_roi * 0.6:
                return 'POINT_RIGHT'
            else:
                return 'POINT_LEFT'  # Default to left
        return 'POINT_LEFT'
    elif finger_count == 1:
        # If very elongated horizontally, it's pointing
        if aspect > 1.3:
            return 'POINT_LEFT'
        return 'FIST'
    else:
        # finger_count == 0: check aspect ratio
        # Fist: compact (aspect close to 1), Point: elongated
        if aspect > 1.4:
            return 'POINT_LEFT'
        return 'FIST'  # Closed fist (no defects, compact shape)
        return 'FIST'  # Closed fist (no defects)

def print_result(label, probs):
    bar_char = '#'
    print(f"\n{'='*50}")
    print(f"  KET QUA: {label}")
    print(f"{'='*50}")
    for cls, prob in zip(CLASSES, probs):
        bar = bar_char * int(prob * 30)
        print(f"  {cls:10s}: {prob*100:5.1f}% [{bar}]")

def draw_gui(frame, label, probs, action, hand_detected):
    display = cv2.resize(frame, (640, 480))

    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (630, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

    color = GESTURE_COLORS.get(label, (255, 255, 255))

    if hand_detected:
        cv2.putText(display, "HAND DETECTED", (20, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(display, "MOVE YOUR HAND", (20, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(display, f"Gesture: {label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(display, f"Action: {action}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    max_prob = max(probs)
    bar_width = int(max_prob * 300)
    cv2.rectangle(display, (20, 120), (320, 140), (50, 50, 50), -1)
    cv2.rectangle(display, (20, 120), (20 + bar_width, 140), color, -1)
    cv2.putText(display, f"{max_prob*100:.1f}%", (330, 138),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

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

    # Try different backends to open camera
    cap = None
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        for idx in [0, 1]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Mo webcam thanh cong: index={idx}, backend={backend}")
                break
            cap.release()
            cap = None
        if cap is not None:
            break

    if cap is None or not cap.isOpened():
        print("Khong mo duoc webcam! Thu tat cac app dung webcam (Zoom, Discord...)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=== Demo Real-time Hand Gesture Recognition ===")
    print("Dang chay... nhan 'q' de thoat")
    print(f"Classes: {CLASSES}")
    print("-" * 50)

    prev_frame = None
    last_print = 0
    interval = 0.5
    hand_count = 0

    # Gesture smoothing - require N consecutive frames to confirm
    confirm_threshold = 8
    gesture_history = []
    confirmed_gesture = 'POINT'
    last_confirmed_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            hand_region, mask = detect_hand_by_motion(frame, prev_frame)
            prev_frame = frame.copy()

            if hand_region is not None:
                x, y, w, h = hand_region
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                roi = frame[y:y+h, x:x+w]

                if roi.size > 0:
                    # Primary: finger-based detection
                    finger_gesture = determine_gesture_by_fingers(roi)

                    # Add to history
                    gesture_history.append(finger_gesture)
                    if len(gesture_history) > 10:
                        gesture_history.pop(0)

                    # Count occurrences
                    gesture_counts = Counter(gesture_history)
                    most_common = gesture_counts.most_common(1)[0][0]

                    # Only change if same gesture appears N times
                    if gesture_counts[most_common] >= confirm_threshold:
                        confirmed_gesture = most_common
                        last_confirmed_time = time.time()

                    label = confirmed_gesture

                    # CNN prediction for display
                    img = cv2.resize(roi, (64, 64))
                    img = img.astype('float32') / 255.0
                    img = img.transpose(2, 0, 1)
                    img = torch.FloatTensor(img).unsqueeze(0)

                    with torch.no_grad():
                        outputs = model(img)
                        probs = torch.softmax(outputs, dim=1)[0].numpy()

                    hand_count += 1
                    hand_detected = True
                else:
                    probs = [0.33, 0.33, 0.34]
                    label = 'POINT'
                    hand_detected = False
            else:
                probs = [0.33, 0.33, 0.34]
                label = 'POINT'
                hand_count = max(0, hand_count - 1)
                hand_detected = hand_count > 2

            action = GESTURE_ACTIONS.get(label, 'UNKNOWN')

            display = draw_gui(frame, label, probs, action, hand_detected)
            cv2.imshow('Hand Gesture Recognition', display)

            current_time = time.time()
            if current_time - last_print >= interval and hand_detected:
                print_result(label, probs)
                print(f"  ACTION: {action}")
                last_print = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nDa thoat!")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera da dong.")

if __name__ == "__main__":
    main()