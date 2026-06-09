# Robot Hand Gesture Recognition - ML Project A2

Project ML nhận dạng cử chỉ tay để điều khiển robot, gồm 3 gestures: WAVE (vẫy tay), FIST (nắm tay), POINT (chỉ tay).

## Cấu trúc project

```
RobotHandsign/
├── dataset/
│   ├── wave/        # Ảnh vẫy tay
│   ├── fist/        # Ảnh nắm tay
│   └── point/       # Ảnh chỉ tay
├── models/
│   └── gesture_classifier.pth  # Model đã train
├── collect_data.py    # Thu thập ảnh từ webcam
├── train_model.py     # Train CNN model
├── demo.py            # Demo real-time
└── README.md
```

## Hướng dẫn chi tiết

### Bước 1: Cài đặt thư viện

```bash
pip install torch torchvision opencv-python numpy scikit-learn
```

Hoặc sử dụng requirements.txt:
```bash
pip install -r requirements.txt
```

### Bước 2: Thu thập dữ liệu

Chạy script thu thập ảnh:
```bash
python collect_data.py
```

**Cách sử dụng:**
- Đặt tay trước webcam với các cử chỉ khác nhau
- Bấm phím `1` = Chụp ảnh WAVE (vẫy tay)
- Bấm phím `2` = Chụp ảnh FIST (nắm tay)
- Bấm phím `3` = Chụp ảnh POINT (chỉ tay)
- Bấm `q` = Thoát

Mỗi loại nên chụp 100-300 ảnh, ở nhiều góc độ khác nhau, ánh sáng khác nhau.

Ảnh sẽ được lưu vào:
- `dataset/wave/` - hình vẫy tay
- `dataset/fist/` - hình nắm tay
- `dataset/point/` - hình chỉ tay

### Bước 3: Train model

Sau khi có dataset, chạy:
```bash
python train_model.py
```

**Thông tin train:**
- Model: CNN (3 layers Conv2D + Dense)
- Image size: 64x64
- Epochs: 30
- Batch size: 32
- Train/Test split: 80/20

**Kết quả train sẽ hiển thị:**
- Loss và Accuracy mỗi epoch
- Test Accuracy cuối cùng
- Confusion Matrix
- Classification Report (Precision, Recall, F1)

**Thông tin model đã train:**
- Dataset: 3000 ảnh (1000 mỗi class: wave, fist, point)
- Test Accuracy: 100%
- Thời gian train: ~2 phút

Model đã train sẵn: `models/gesture_classifier.pth`

**Nếu muốn train lại với dataset mới:**
```bash
python train_model.py
```

### Bước 4: Chạy demo real-time

```bash
python demo.py
```

**Màn hình sẽ hiển thị:**
- Camera sẽ hiển thị theo realtime
- Mỗi 0.5 giây sẽ in kết quả nhận diện
- Kết quả bao gồm: Loại cử chỉ + Confidence %
- Bấm `q` để thoát

**Output ví dụ:**
```
==================================================
  KET QUA: WAVE
==================================================
  WAVE       :  85.3% [####################]
  FIST       :  10.2% [#####]
  POINT      :   4.5% [##]
```

**Lưu ý:**
- Phải tải file hand_landmarker.task về cùng đường dẫn với file demo.py

  
## Giải quyết vấn đề thường gặp

### Lỗi webcam không mở được
- Kiểm tra webcam có bị cắm bởi app khác không
- Thử ngắt cáp USB và cắm lại
- Kiểm tra quyền truy cập camera

### Lỗi Unicode trên Windows
- Script đã cấu hình sẵn UTF-8
- Nếu vẫn lỗi, thiết lập terminal encoding:
  ```bash
  chcp 65001
  ```

### Model không chính xác
- Thu thập nhiều ảnh hơn (300+ mỗi loại)
- Chụp ảnh ở nhiều góc, ánh sáng khác nhau
- Loại bỏ ảnh mờ, ảnh quá tối
- Tăng số epochs (50-100) trong train_model.py

### Không load được model
- Kiểm tra file `models/gesture_classifier.pth` có tồn tại không
- Chạy lại train_model.py nếu file bị lỗi

## Thông số kỹ thuật

| Thông số | Giá trị |
|----------|---------|
| Image size | 64x64 pixels |
| Classes | 3 (wave, fist, point) |
| Model format | PyTorch (.pth) |
| Input | RGB image |
| Output | Class probability (0-1) |

## Cấu hình nâng cao (tùy chọn)

### Tăng số epochs
Trong file `train_model.py`, thay đổi:
```python
for epoch in range(50):  # tăng từ 30 lên 50
```

### Tăng độ phân giải ảnh
Thay đổi `IMG_SIZE = 128` trong tất cả các file.

### Thêm lớp CNN
Chỉnh sửa `CNN` class trong `train_model.py`:
```python
nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
```
