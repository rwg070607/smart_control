import cv2
import mediapipe as mp
import numpy as np
import pyautogui  # 앱 전환 및 클릭을 위해 추가

# 얼굴 추적 초기화
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 클릭 및 스크롤 상태 변수 
scroll_timer = 0
scroll_threshold = 5  # 스크롤 감도
safety_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)

    if face_results.multi_face_landmarks:
        safety_mode = False  # 얼굴이 감지되면 안전 모드 해제

        for face_landmarks in face_results.multi_face_landmarks:
            # 랜드마크 표시
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

            # 왼쪽 눈 중앙 위치
            left_eye_center = (int(face_landmarks.landmark[33].x * frame.shape[1]),
                               int(face_landmarks.landmark[33].y * frame.shape[0]))

            # 스크롤 기능
            if left_eye_center[1] < frame.shape[0] // 3:
                scroll_timer += 1
            elif left_eye_center[1] > 2 * frame.shape[0] // 3:
                scroll_timer -= 1
            else:
                scroll_timer = 0

            if scroll_timer > scroll_threshold * 10:
                pyautogui.scroll(10)  
                print("Scroll Up")
                scroll_timer = 0
            elif scroll_timer < -scroll_threshold * 10:
                pyautogui.scroll(-10)  
                print("Scroll Down")
                scroll_timer = 0

    else:
        safety_mode = True  # 얼굴이 감지되지 않으면 안전 모드 활성화

    # 안전 모드 메시지 출력 및 검정색 화면
    if safety_mode:
        print("Entering Safe Mode")
        frame = np.zeros_like(frame)  # 화면을 검정색으로 변경
        cv2.putText(frame, "Safe Mode Activated", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 결과 출력
    cv2.imshow("Eye Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()