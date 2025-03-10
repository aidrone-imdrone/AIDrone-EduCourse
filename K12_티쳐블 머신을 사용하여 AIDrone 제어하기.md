
# Control AIDrone Using Teachable Machine

### 1) Start Teachable Machine

<br/>

<img src="https://github.com/user-attachments/assets/766422b4-08dc-457e-94e9-928b495263ae" width="800">

<br/>

###  Make Class 5  -> TakeOff, Landing, Up, Down, Left, Right, BackGround, Hovering

![image](https://user-images.githubusercontent.com/122161666/227949247-011c49d0-adc2-4404-80d6-54166aa11d8a.png)

<br/>

![image](https://github.com/user-attachments/assets/efb0f62b-25c1-4111-a6c9-600c9eeb2247)

-**백그라운드에 다양한 환경을 찍어 주시기를 바랍니다.**

-**각 클래스당 900 ~ 1000장 찍음**

<br/>

![image](https://github.com/user-attachments/assets/b482bf37-10c3-4259-8f37-9b55db6d4cba)

<br/>

### After take direction Picture all and then Click Learning Model, If finished Learning Model, Click exporting the model 

<br/>

![image](https://user-images.githubusercontent.com/122161666/227953634-227318ce-7fee-40fe-a35e-26b4e10d64fd.png)

<br/>

### When download Model from Teachable Machine, it has two file inside(model_unquant.tflite and labels.txt)

<br/>

<img src="https://github.com/user-attachments/assets/7662ce9d-983e-48cd-ba87-a1f14a9e7e77" width="800">

<br/><br/>

위 예제 TFLite Model Download Link :  https://drive.google.com/file/d/1plJbt3rtZM8ykWdG4hcWjcVjJFqcHOpI/view?usp=drive_link

위 예제 Tensorflow Keras Model Download Link : https://drive.google.com/file/d/1-xWnTJtOaQ46F639xwqL1jQOYbWGgBrE/view?usp=drive_link

<br/>

### 2) Tensorflow Lite 라이브러리를 설치한다. 

<br/>

       sudo apt install && sudo apt upgrade -y 
       pip  install  tflite-runtime
       pip  install  numpy

<br/>

### 3)  Samba를 통해  TFLite 모델파일과 label.txt 파일을 전송한다. 

<br/>

### 4)  Code (gesture_drone_contorl_pi.py)

<br/>

    import cv2
    import numpy as np
    import tensorflow as tf
    from time import sleep, time
    from pyaidrone.aiDrone import *
    from pyaidrone.deflib import *
    from threading import Thread
    import os
    
    # 전역 변수
    Height = 100
    ready = -1
    tof = 0
    running = True
    current_gesture = "NONE"
    confidence_threshold = 0.70  # 제스처 인식 신뢰도 임계값
    camera_index = 0  # 카메라 인덱스, 필요시 변경
    
    # 라벨 파일에서 클래스 이름 로드하는 함수
    def load_labels(label_file):
        with open(label_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    # 드론 명령 매핑
    def execute_drone_command(aidrone, gesture):
        global Height
        
        if gesture == "TakeOff":  # 이륙 (엄지척)
            print("이륙!")
            aidrone.takeoff()
        
        elif gesture == "Landing":  # 착륙 (손 내리기)
            print("착륙!")
            aidrone.landing()
        
        elif gesture == "Left":  # 왼쪽 (왼쪽 손 제스처)
            print("왼쪽!")
            aidrone.velocity(LEFT, 50)
            sleep(0.2)
            aidrone.velocity(LEFT, 0)
        
        elif gesture == "Right":  # 오른쪽 (오른쪽 손 제스처)
            print("오른쪽!")
            aidrone.velocity(RIGHT, 50)
            sleep(0.2)
            aidrone.velocity(RIGHT, 0)
        
        elif gesture == "Hovering":  # 호버링 (특정 손 제스처)
            print("정지/호버링!")
            aidrone.velocity(FRONT, 0)
            aidrone.velocity(RIGHT, 0)
    
        # Background 클래스는 무시 (아무 동작 안함)
    
    # 티쳐블 머신에서 내보낸 모델 로드 함수
    def load_teachable_machine_model(model_path):
        # TensorFlow Lite 모델 로드
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 입력 및 출력 정보 가져오기
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return interpreter, input_details, output_details
    
    # 이미지 전처리 (티쳐블 머신 모델에 맞게 조정)
    def prepare_image(img, input_details):
        input_shape = input_details[0]['shape']
        
        # 입력 크기에 맞게 이미지 리사이즈
        img = cv2.resize(img, (input_shape[1], input_shape[2]))
        
        # 필요한 경우 색상 채널 변환 (RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 정규화
        img = img / 255.0
        
        # 배치 차원 추가
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        return img
    
    # 제스처 감지 스레드 함수
    def gesture_detection_thread(model_path, label_path):
        global current_gesture, running
        
        # 클래스 이름 로드
        class_names = load_labels(label_path) if os.path.exists(label_path) else ["TakeOff", "Landing", "Left", "Right", "Background", "Hovering"]
        print(f"로드된 클래스: {class_names}")
        
        # 카메라 설정
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("카메라를 열 수 없습니다!")
            running = False
            return
        
        # 모델 로드
        try:
            interpreter, input_details, output_details = load_teachable_machine_model(model_path)
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            cap.release()
            running = False
            return
        
        # 연속 인식 처리를 위한 변수
        last_gesture = "NONE"
        gesture_counter = 0
        required_consistent_frames = 3  # 연속 인식 프레임 수
        
        print("제스처 인식 시작!")
        
        while running:
            # 프레임 캡처
            ret, frame = cap.read()
            if not ret:
                print("카메라에서 프레임을 읽을 수 없습니다!")
                break
            
            # 이미지 전처리
            processed_img = prepare_image(frame, input_details)
            
            # 모델 추론
            interpreter.set_tensor(input_details[0]['index'], processed_img)
            interpreter.invoke()
            
            # 결과 얻기
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # 가장 높은 확률의 클래스 찾기
            max_index = np.argmax(predictions)
            confidence = predictions[max_index]
            
            # Background 클래스는 무시
            if "Background" in class_names and class_names[max_index] == "Background":
                predicted_gesture = "NONE"
            # 충분한 신뢰도가 있는 경우에만 제스처 인식
            elif confidence > confidence_threshold:
                predicted_gesture = class_names[max_index]
            else:
                predicted_gesture = "NONE"
            
            # 연속 프레임 처리 (노이즈 필터링)
            if predicted_gesture == last_gesture:
                gesture_counter += 1
            else:
                gesture_counter = 0
                last_gesture = predicted_gesture
            
            # 충분한 연속 프레임이 감지되면 전역 제스처 업데이트
            if gesture_counter >= required_consistent_frames and predicted_gesture != "NONE":
                if current_gesture != predicted_gesture:
                    current_gesture = predicted_gesture
                    print(f"제스처 감지: {current_gesture} (신뢰도: {confidence:.2f})")
            
            # 화면에 결과 표시 (디버깅용)
            label = f"{predicted_gesture if predicted_gesture != 'NONE' else class_names[max_index]}: {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
        
        # 자원 해제
        cap.release()
        cv2.destroyAllWindows()
    
    # 드론 제어 함수
    def drone_control(aidrone):
        global current_gesture, running
        
        last_executed_gesture = "NONE"
        command_cooldown = 1.0  # 명령 간 쿨다운 시간 (초)
        last_command_time = time() - command_cooldown
        
        while running:
            # 현재 시간
            current_time = time()
            
            # 새로운 제스처가 감지되고 쿨다운 시간이 지난 경우
            if (current_gesture != "NONE" and 
                current_gesture != last_executed_gesture and 
                (current_time - last_command_time) > command_cooldown):
                
                # 드론 명령 실행
                execute_drone_command(aidrone, current_gesture)
                
                # 마지막 실행 제스처 및 시간 업데이트
                last_executed_gesture = current_gesture
                last_command_time = current_time
            
            # 짧은 대기 시간 (CPU 사용량 감소)
            sleep(0.1)
    
    # 메인 함수
    def main():
        global running
        
        # 드론 연결
        try:
            aidrone = AIDrone()
            aidrone.Open("/dev/serial0")  # 필요에 따라 포트 변경
            aidrone.setOption(0)
            sleep(0.5)
            print("드론에 연결되었습니다!")
        except Exception as e:
            print(f"드론 연결 오류: {e}")
            return
        
        # 티쳐블 머신 모델 및 라벨 경로
        model_path = "model_unquant.tflite"  # 모델 파일 경로 수정
        label_path = "labels.txt"  # 라벨 파일 경로
        
        try:
            # 제스처 감지 스레드 시작
            detection_thread = Thread(target=gesture_detection_thread, args=(model_path, label_path))
            detection_thread.daemon = True
            detection_thread.start()
            
            # 드론 제어 시작
            drone_control(aidrone)
        
        except KeyboardInterrupt:
            print("프로그램 종료 요청...")
        finally:
            # 프로그램 종료 처리
            running = False
            
            # 드론 안전 착륙
            print("드론 착륙 중...")
            aidrone.landing()
            sleep(2)
            
            # 드론 연결 종료
            aidrone.Close()
            print("프로그램이 안전하게 종료되었습니다.")
    
    if __name__ == "__main__":
        main()


<br/><br/>

### 5) Code 2 (gesture_drone_cotnrol_pc.py)

<br/>

    from time import sleep
    import cv2
    import numpy as np
    import tensorflow as tf
    from pyaidrone.aiDrone import *
    from pyaidrone.deflib import *
    from pyaidrone.ikeyevent import *
    import sys
    
    # 전역 변수 설정
    Height = 100
    ready = -1
    tof = 0
    last_gesture = "Background"
    gesture_confirmed = False
    confirmation_count = 0
    CONFIRMATION_THRESHOLD = 5  # 제스처 확인을 위한 연속 프레임 수
    
    # 라벨 로드
    def load_labels(label_file):
        with open(label_file, 'r') as f:
            labels = {}
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    labels[int(parts[0])] = parts[1]
        return labels
    
    # 모델 로드
    def load_model(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    
    # 드론으로부터 데이터 수신
    def receive_data(packet):
        global ready
        global tof
    
        ready = packet[7] & 0x03
        if packet[7] & 0x20 == 0x20 and len(packet) > 13:
            tof = packet[13]
            print("tof : ", tof)
    
    # 제스처 인식 및 드론 제어
    def main():
        global Height, last_gesture, gesture_confirmed, confirmation_count
        
        # 웹캠 초기화
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return
        
        # 모델 및 라벨 로드
        try:
            model = load_model("gesture_model.h5")
            labels = load_labels("labels.txt")
            print("모델과 라벨이 로드되었습니다.")
        except Exception as e:
            print(f"모델 또는 라벨 로드 실패: {e}")
            return
        
        # 드론 초기화
        aidrone = AIDrone()
        ikey = IKeyEvent()
        
        try:
            aidrone.Open("COM3")  # 포트 설정
            aidrone.setOption(0)
            sleep(0.5)
            print("드론 연결 성공")
        except Exception as e:
            print(f"드론 연결 실패: {e}")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        # 메인 루프
        while True:
            # 키보드 입력 확인 (비상 정지 등)
            if ikey.isKeyEscPressed():
                print("프로그램 종료")
                break
            
            # 웹캠에서 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # 이미지 전처리
            resized_frame = cv2.resize(frame, (224, 224))
            normalized_frame = resized_frame / 255.0
            input_data = np.expand_dims(normalized_frame, axis=0)
            
            # 제스처 예측
            predictions = model.predict(input_data)
            gesture_id = np.argmax(predictions[0])
            confidence = predictions[0][gesture_id]
            
            current_gesture = labels.get(gesture_id, "Unknown")
            
            # 제스처 확인 (노이즈 방지를 위해 여러 프레임 연속으로 같은 제스처인지 확인)
            if current_gesture == last_gesture and confidence > 0.7:
                confirmation_count += 1
            else:
                confirmation_count = 0
                last_gesture = current_gesture
            
            # 제스처가 확인되면 드론 제어
            if confirmation_count >= CONFIRMATION_THRESHOLD and not gesture_confirmed:
                gesture_confirmed = True
                
                print(f"제스처 감지: {current_gesture} (신뢰도: {confidence:.2f})")
                
                # 제스처별 드론 동작 제어
                if current_gesture == "TakeOff":
                    print("이륙")
                    aidrone.takeoff()
                
                elif current_gesture == "Landing":
                    print("착륙")
                    aidrone.landing()
                
                elif current_gesture == "Left":
                    print("왼쪽")
                    aidrone.velocity(LEFT, 100)
                    sleep(1)
                    aidrone.velocity(LEFT, 0)
                
                elif current_gesture == "Right":
                    print("오른쪽")
                    aidrone.velocity(RIGHT, 100)
                    sleep(1)
                    aidrone.velocity(RIGHT, 0)
                
                elif current_gesture == "Hovering":
                    print("호버링")
                    aidrone.velocity(FRONT, 0)
                    aidrone.velocity(RIGHT, 0)
            
            # 일정 시간 동안 같은 제스처가 유지되지 않으면 제스처 확인 상태 초기화
            elif current_gesture != last_gesture:
                gesture_confirmed = False
            
            # 화면에 정보 표시
            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 표시
            cv2.imshow("Gesture Drone Control", frame)
            
            # q 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 정리
        aidrone.landing()  # 안전을 위해 착륙
        sleep(1)
        aidrone.Close()
        cap.release()
        cv2.destroyAllWindows()
    
    if __name__ == "__main__":
        main()
  
