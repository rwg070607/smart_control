import cv2

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()

    # 텍스트 추가
    text = "Hello, OpenCV!"
    org = (50, 50)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    lineType = cv2.LINE_AA

    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, lineType)

    # 프레임 보여주기
    cv2.imshow("Webcam", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()