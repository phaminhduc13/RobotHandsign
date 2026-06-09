import cv2
import os
import time
import sys



# Cấu hình
IMG_SIZE = 64
SAVE_DIRS = {
    ord('1'): 'dataset/wave',
    ord('2'): 'dataset/fist',
    ord('3'): 'dataset/point'
}

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong mo duoc webcam!")
        return

    print("=== Thu thap du lieu cu chi tay ===")
    print("Ban 1: Lưu WAVE (vay tay)")
    print("Ban 2: Lưu FIST (nam tay)")
    print("Ban 3: Lưu POINT (chi tay)")
    print("Ban 'q': Thoat")
    print()

    counters = {d: 0 for d in SAVE_DIRS.values()}

    # Đếm ảnh hiện có
    for d in SAVE_DIRS.values():
        if os.path.exists(d):
            counters[d] = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize và hiển thị
        preview = cv2.resize(frame, (320, 240))
        cv2.putText(preview, "1:Wave 2:Fist 3:Point | q:Quit", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i, (key, dir_path) in enumerate(SAVE_DIRS.items()):
            label = dir_path.split('/')[-1].upper()
            cv2.putText(preview, f"{chr(key)}: {counters[dir_path]} {label}",
                        (10, 45 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow('Thu tap anh cu chi tay', preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in SAVE_DIRS:
            dir_path = SAVE_DIRS[key]
            os.makedirs(dir_path, exist_ok=True)

            # Lưu ảnh đã resize
            small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            filename = f"{dir_path}/img_{int(time.time()*1000)}.jpg"
            cv2.imwrite(filename, small)
            counters[dir_path] += 1
            print(f"Da luu: {filename} ({counters[dir_path]} anh)")

    cap.release()
    cv2.destroyAllWindows()
    print("Xong!")

if __name__ == "__main__":
    main()
