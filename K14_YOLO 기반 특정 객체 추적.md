# **YOLO 기반 특정 객체 추적 프로젝트**

## **1. 개요**
### **프로젝트 목표**
이 프로젝트에서는 **YOLO (You Only Look Once) 기반으로 특정 객체를 실시간으로 추적하는 드론 시스템**을 구현합니다. YOLO 모델을 사용하여 카메라 스트림에서 특정 객체(예: 사람, 자동차 등)를 감지하고, 사용자가 직접 선택한 객체를 따라가도록 합니다.

### **필요한 기술 및 라이브러리**
- **Python**: 프로그래밍 언어
- **OpenCV**: 실시간 영상 처리
- **YOLOv4/v5**: 객체 탐지 모델
- **pyaidrone**: 드론 제어 라이브러리
- **NumPy**: 데이터 처리 및 배열 연산
- **Raspberry Pi Zero 2W**: 드론의 연산 장치
- **Camera Module**: 영상 데이터 수집

### **기능 개요**
- **YOLO를 이용한 실시간 객체 탐지**
- **사용자가 선택한 객체를 추적**
- **드론이 객체의 위치를 추적하여 이동**
- **YOLO에 없는 객체를 추가 학습하여 인식 가능**

---

## **2. 환경 설정**
### **필요한 패키지 설치**
```bash
pip install opencv-python numpy torch torchvision pyyaml pyaidrone
```

### **YOLO 모델 다운로드**
[YOLOv5 공식 저장소](https://github.com/ultralytics/yolov5)에서 모델을 다운로드합니다.
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

---

## **3. YOLO에 새로운 객체 추가하기**
YOLO 기본 모델에 없는 새로운 객체를 학습하려면 다음 단계를 수행합니다.

### **1단계: 데이터 수집**
- 직접 카메라로 다양한 각도에서 객체 사진을 촬영하거나 공개 데이터셋을 활용합니다. (K13 예제 참조)
- 이미지를 `dataset/images/` 폴더에 저장합니다.
- 충분한 데이터 수집이 중요하며, 다양한 환경(조명, 각도, 배경)에서 촬영된 이미지가 필요합니다.

### **2단계: 데이터 전처리 (이미지 크기 조정)**
- YOLO 모델의 입력 사이즈와 일관성을 유지하기 위해 이미지를 640x640 픽셀로 조정합니다.
```python
import cv2
import os

dataset_path = "dataset/images/"
output_path = "dataset/resized/"
os.makedirs(output_path, exist_ok=True)

for file in os.listdir(dataset_path):
    img = cv2.imread(os.path.join(dataset_path, file))
    img_resized = cv2.resize(img, (640, 640))
    cv2.imwrite(os.path.join(output_path, file), img_resized)
```

### **3단계: 데이터 라벨링**
- YOLO 모델이 객체를 학습하려면 라벨링된 데이터가 필요합니다.
- [LabelImg](https://github.com/heartexlabs/labelImg) 툴을 설치하여 이미지에 라벨을 지정합니다.
```bash
pip install labelImg
labelImg
```
- `LabelImg`를 실행하여 `dataset/resized/` 폴더 내의 이미지를 불러옵니다.
- 객체를 박스로 지정하고 이름을 설정한 후, YOLO 형식(`.txt`)으로 저장합니다.
- 생성된 `.txt` 파일은 각 이미지와 동일한 이름을 가지며, 라벨된 좌표 정보를 포함합니다.

### **4단계: YOLO 학습 데이터 구성**
- YOLO는 학습을 위해 `train`, `val` 데이터셋을 필요로 합니다.
- 아래와 같은 폴더 구조를 유지해야 합니다.
```
yolo_custom_dataset/
  ├── images/
  │   ├── train/   # 학습 이미지
  │   ├── val/     # 검증 이미지
  ├── labels/
  │   ├── train/   # 학습 라벨
  │   ├── val/     # 검증 라벨
```
- 수집한 데이터의 `80%`를 `train/`, `20%`를 `val/` 폴더에 저장합니다.

- `custom_data.yaml` 파일을 생성하여 학습을 위한 설정을 정의합니다.
```yaml
train: /path_to_dataset/yolo_custom_dataset/images/train
val: /path_to_dataset/yolo_custom_dataset/images/val
nc: 1  # 객체 개수
names: ['custom_object']  # 객체 이름
```

### **5단계: YOLO 모델 학습**
YOLO 모델을 학습하여 새로운 객체를 감지하도록 설정합니다.
```bash
python train.py --img 640 --batch 16 --epochs 50 --data custom_data.yaml --weights yolov5s.pt
```
- `--img 640`: 이미지 크기
- `--batch 16`: 배치 크기 (RAM에 따라 조정 가능)
- `--epochs 50`: 학습 반복 횟수
- `--data custom_data.yaml`: 데이터셋 설정 파일 경로
- `--weights yolov5s.pt`: 사전 학습된 YOLO 모델 사용

- 학습이 완료되면 모델은 `runs/train/exp/weights/best.pt`에 저장됩니다.

### **6단계: 학습된 모델 테스트**
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source test.jpg
```
- 학습된 모델을 사용하여 새로운 이미지에서 객체를 감지할 수 있습니다.

```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source video.mp4
```
- 실시간 영상에서 YOLO가 객체를 감지하는 기능을 테스트할 수 있습니다.

### **7단계: 모델을 YOLO와 통합하여 객체 추적 기능 구현**
Python 코드에서 학습된 모델을 로드하여 사용합니다.

```python
import torch
import cv2
import numpy as np

# 학습된 YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt', source='local')

# 이미지 로드
detect_img = cv2.imread('test.jpg')
results = model(detect_img)

# 결과 표시
results.show()
```


## **4. 마우스로 선택한 객체 추적 기능 추가**
YOLO 모델을 활용하여 탐지된 객체를 마우스로 선택하여 추적하는 기능을 추가합니다.

```python
import torch
import cv2
import numpy as np
from pyaidrone.aiDrone import *
from pyaidrone.deflib import *

# YOLO 모델 로드 (사용자 학습 모델 적용 가능)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='custom_model.pt', source='local')

# 드론 초기화
aidrone = AIDrone()
aidrone.Open("COM3")

# 카메라 설정
cap = cv2.VideoCapture(0)
selected_object = None
selected_bbox = None

def select_object(event, x, y, flags, param):
    global selected_object, selected_bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if x1 < x < x2 and y1 < y < y2:
                selected_object = model.names[int(cls)]
                selected_bbox = (x1, y1, x2, y2)
                print(f"Selected Object: {selected_object}")
                break

cv2.namedWindow("YOLO Object Tracking")
cv2.setMouseCallback("YOLO Object Tracking", select_object)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()
    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
    target_x, target_y = None, None

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = model.names[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if selected_object and label == selected_object:
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2

    if target_x and target_y:
        if target_x < center_x - 50:
            aidrone.velocity(LEFT, 50)  # 왼쪽 이동
        elif target_x > center_x + 50:
            aidrone.velocity(RIGHT, 50)  # 오른쪽 이동
        else:
            aidrone.velocity(FRONT, 50)  # 전진
    else:
        aidrone.velocity(FRONT, 0)  # 멈춤

    cv2.imshow('YOLO Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
aidrone.Close()
```

이제 마우스로 클릭하여 원하는 객체를 선택하면, 드론이 해당 객체를 따라가도록 동작하며, 새로운 객체를 학습하여 추가할 수도 있습니다.

---

## **5. 프로젝트 요약**
### **핵심 기능**
- **YOLO 기반 객체 감지**
- **사용자가 선택한 특정 객체를 추적**
- **YOLO에 없는 객체를 추가 학습하여 추적 가능**
- **드론이 객체를 따라가도록 제어**

### **확장 가능성**
- **추적 정확도 향상 (YOLOv7 등 최신 모델 적용)**
- **다양한 객체 탐지 및 추적 기능 추가**
- **강화학습을 활용한 최적 경로 학습**
- **GPS 및 LiDAR 센서와 결합하여 실외에서도 활용 가능**

이 프로젝트를 통해 YOLO를 이용한 객체 감지 및 드론의 자동 추적을 구현할 수 있습니다. 🚀




