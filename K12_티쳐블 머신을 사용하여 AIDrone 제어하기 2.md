# 라즈베리파이 카메라 기반 드론 제어 시스템

이 프로젝트는 라즈베리파이 카메라를 사용하여 손동작을 인식하고, 인식된 손동작을 통해 드론을 제어하는 시스템입니다. TensorFlow Lite 모델을 사용하여 경량화된 환경에서도 효율적으로 실행됩니다.

## 기능

- 라즈베리파이 카메라를 통한 실시간 손동작 인식
- TensorFlow Lite 모델을 활용한 효율적인 추론
- 90% 이상의 신뢰도를 가진 손동작만 인식
- 같은 손동작이 1초 이상 유지될 때만 드론 명령 실행
- 다양한 손동작을 통해 드론 제어 (이륙, 착륙, 좌/우 이동, 호버링)

## 요구사항

- 라즈베리파이 (3 이상 권장)
- 라즈베리파이 카메라 모듈
- 호환 가능한 드론 (aiDrone 라이브러리 지원 모델)
- 다음 Python 패키지:
  - tflite_runtime
  - picamera2
  - opencv-python
  - numpy
  - pyaidrone (드론 제어용 라이브러리)

## 설치 방법

```bash
# 필요한 패키지 설치
pip install tflite_runtime
pip install picamera2
pip install opencv-python
pip install numpy

# pyaidrone 라이브러리 설치 (드론 모델에 맞는 라이브러리를 설치하세요)
# 예: pip install pyaidrone
```

## 사용 방법

1. TensorFlow Lite 모델과 라벨 파일 준비
   - Teachable Machine을 사용하여 손동작 모델을 생성하고 TFLite 모델로 내보내기
   - 모델 파일(`model.tflite`)과 라벨 파일(`labels.txt`)을 프로젝트 디렉토리에 위치

<br/>

<img src="https://github.com/user-attachments/assets/e6a6ef49-487f-4f16-866d-165b9f7d097f"  width="800">

<br/><br/>

-**위 예제의 결과인 Tensorflow Lite Model Download Link** 

  https://drive.google.com/file/d/1as4y0XBxWTalh1ZJsO-hn06-9TRT9Kvw/view?usp=drive_link

<br/>

2. 코드 실행
   ```bash
   python gesture_drone_control_pi.py
   ```
- **tflife 폴더 안에 있음**

<br/>

3. 손동작으로 드론 제어
   - "TakeOff": 드론 이륙
   - "Landing": 드론 착륙
   - "Left": 드론 왼쪽으로 이동
   - "Right": 드론 오른쪽으로 이동
   - "Hovering": 드론 제자리 호버링

<br/>

## 코드 설명

### 주요 구성 요소

- **카메라 처리**: Picamera2 라이브러리를 사용하여 라즈베리파이 카메라에서 이미지를 가져옵니다.
- **모델 로딩**: TFLite 인터프리터를 사용하여 모델을 로드하고 추론을 수행합니다.
- **손동작 인식**: 카메라 입력을 전처리하고 모델을 통해 손동작을 인식합니다.
- **지속 시간 체크**: 같은 손동작이 1초 이상 지속될 때만 명령을 실행합니다.
- **드론 제어**: 인식된 손동작에 따라 드론에 명령을 전송합니다.

### 핵심 알고리즘

```python
# 새로운 제스처인지 확인
if current_gesture != last_gesture:
    gesture_start_time = time()  # 현재 시간으로 시작 시간 설정
    last_gesture = current_gesture
    
# 제스처가 필요한 시간(1초) 동안 유지되었는지 확인
elif time() - gesture_start_time >= gesture_duration_threshold:
    # 드론 명령 실행
    # ...
```

이 알고리즘은 사용자가 손동작을 변경할 때마다 타이머를 리셋하고, 같은 손동작이 1초 이상 유지될 때만 드론 명령을 실행합니다. 이를 통해 우발적인 인식 오류로 인한 잘못된 명령 실행을 방지합니다.

## 커스터마이징

- **포트 설정**: 드론이 연결된 포트를 환경에 맞게 수정 (`aidrone.Open("COM3")` 부분)  =>  aidrone.Open("/dev/serial0")
- **신뢰도 임계값**: 필요에 따라 신뢰도 임계값 조정 (`confidence_threshold = 90`)
- **지속 시간**: 손동작 유지 시간 조정 (`gesture_duration_threshold = 1.0`)
- **추가 제스처**: 더 많은 손동작과 드론 명령을 추가하여 기능 확장 가능

## 문제 해결

- **카메라 오류**: 카메라 연결 및 설정 확인
- **모델 로딩 실패**: 모델 및 라벨 파일 경로 확인
- **드론 연결 오류**: 드론 배터리 및 연결 포트 확인
- **낮은 인식률**: 더 많은 훈련 데이터로 모델 재훈련 또는 조명 조건 개선


