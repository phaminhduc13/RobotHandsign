import cv2
import numpy as np
import os
import sys

# Fix Unicode for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

IMG_SIZE = 64
NUM_IMAGES = 100

def generate_wave():
    """Wave - vẫy tay - hình chữ V hoặc lòng bàn tay"""
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype='uint8') * 200
    # Vẽ hình bàn tay đơn giản
    center = IMG_SIZE // 2
    x = center + np.random.randint(-8, 8)
    y = center + np.random.randint(-8, 8)
    # Thân bàn tay
    color = (np.random.randint(80, 180), np.random.randint(50, 120), np.random.randint(30, 100))
    # Vẽ các ngón tay
    for i in range(5):
        angle = -0.6 + i * 0.3 + np.random.uniform(-0.1, 0.1)
        length = np.random.randint(12, 22)
        end_x = int(x + length * np.cos(angle))
        end_y = int(y - length * np.sin(angle))
        cv2.line(img, (x, y), (end_x, end_y), color, 3)
    return img

def generate_fist():
    """Fist - nắm tay - hình khối tròn"""
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype='uint8') * 200
    center = IMG_SIZE // 2
    x = center + np.random.randint(-8, 8)
    y = center + np.random.randint(-8, 8)
    radius = np.random.randint(14, 22)
    # Clamp
    x = max(radius, min(x, IMG_SIZE - radius))
    y = max(radius, min(y, IMG_SIZE - radius))
    color = (np.random.randint(80, 180), np.random.randint(50, 120), np.random.randint(30, 100))
    cv2.circle(img, (x, y), radius, color, -1)
    # Thêm texture
    for _ in range(5):
        px = x + np.random.randint(-radius//2, radius//2)
        py = y + np.random.randint(-radius//2, radius//2)
        cv2.circle(img, (px, py), 2, (color[0]//3, color[1]//3, color[2]//3), -1)
    return img

def generate_point():
    """Point - chỉ tay - một ngón dài"""
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype='uint8') * 200
    center = IMG_SIZE // 2
    x = center + np.random.randint(-8, 8)
    y = center + np.random.randint(-8, 8)
    length = np.random.randint(18, 28)
    angle = np.random.uniform(-0.3, 0.3)
    end_x = int(x + length * np.cos(angle))
    end_y = int(y - length * np.sin(angle))
    # Clamp
    end_x = max(5, min(end_x, IMG_SIZE - 5))
    end_y = max(5, min(end_y, IMG_SIZE - 5))
    color = (np.random.randint(80, 180), np.random.randint(50, 120), np.random.randint(30, 100))
    cv2.line(img, (x, y), (end_x, end_y), color, 4)
    # Thêm bàn tay nhỏ
    cv2.circle(img, (x, y), 8, color, -1)
    return img

def main():
    os.makedirs('dataset/wave', exist_ok=True)
    os.makedirs('dataset/fist', exist_ok=True)
    os.makedirs('dataset/point', exist_ok=True)

    generators = {
        'wave': generate_wave,
        'fist': generate_fist,
        'point': generate_point
    }

    for name, gen_func in generators.items():
        print(f"Tao anh {name}...")
        for i in range(NUM_IMAGES):
            img = gen_func()
            filename = f"dataset/{name}/syn_{i+1:04d}.jpg"
            cv2.imwrite(filename, img)
        print(f"  Da tao {NUM_IMAGES} anh {name}")

    print("\nXong! Chay python train_model.py de train model.")

if __name__ == "__main__":
    main()