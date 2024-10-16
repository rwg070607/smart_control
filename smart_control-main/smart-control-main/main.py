import cv2
import mediapipe as mp
from PIL import Image, ImageDraw
import pystray
import time
import threading
import pyautogui
import math as mt
#손가락 위치
fig_position = [0 for i in range(21)]
#인식 손가락과 거리
fig_distance = [0 for i in range(21)]
#onoff
tracking_on = False
#action 기준 손가락
main_fig = 8
#인식 손가락
mouse_fig = 9
#화면 전체 사이즈
screen_width, screen_height = pyautogui.size()
# 비율
def windowsize(fullsize,a):
    return (a-0.2)*(fullsize/0.7)

def distance(main_fig,position):
    try:
        x=position[main_fig][0]
        y=position[main_fig][1]
        for idx in range(len(position)):
            position[idx] = [x-position[idx][0],y-position[idx][1]]
    except:
        pass
    return position

def mouse_position(position,idx):
    try:
        if position[idx][0]-0.2<0:
            pyautogui.moveTo(0, round(windowsize(screen_height,position[idx][1]),3), duration=0.1)
        elif position[idx][1]-0.2<0:
            pyautogui.moveTo(round(windowsize(screen_width,position[idx][0]),3), 0, duration=0.1)
        else:
            pyautogui.moveTo(round(windowsize(screen_width,position[idx][0]),3), round(windowsize(screen_height,position[idx][1]),3), duration=0.1)
    except:
        print("can't move")
        pass

def fig_text(frame,fig_idx,position):
    text = str(fig_idx)
    try:
        org = (int(position[fig_idx][0]*640), int(position[fig_idx][1]*480))
    except:
        org=(0,0)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    lineType = cv2.LINE_AA

    cv2.putText(frame, text, org, fontFace, fontScale, color, thickness, lineType)
# 아이콘에 사용할 이미지 생성 함수
def create_image():
    # 64x64 크기의 빈 이미지를 생성하고, 그 위에 사각형을 그린다.
    image = Image.open("imgs/icon.png")
    return image

# 프로그램을 종료하는 함수
def on_quit(icon, item):
    icon.stop()  # 아이콘을 트레이에서 제거하고 프로그램 종료
#leftclick
def click(option):
    notloop = True
    if option[0][0] & notloop:
        try:
            pyautogui.click()
            notloop = False
        except:
            pass
    elif option[1][0]:
        notloop = True
        pass
    

#main
def background_task():
    # MediaPipe 손 객체 초기화
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils  # 손 랜드마크 그리기 도구

    # 비디오 캡처 객체 생성 (웹캠 사용)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 손 추적 모델 초기화
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    while cap.isOpened():
        success, image = cap.read()  # 웹캠으로부터 프레임 읽기
        image =  cv2.flip(image, 1)
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            break

        # 성능 향상을 위해 이미지를 쓰기 불가능 모드로 변환
        image.flags.writeable = False
        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 손 인식 수행
        results = hands.process(image)

        # 이미지 다시 쓰기 가능 모드로 변환
        image.flags.writeable = True
        # RGB 이미지를 다시 BGR로 변환
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 손이 인식되었을 때
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 각 손 랜드마크에 대한 처리
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # 랜드마크의 x, y, z 좌표를 가져옴
                    fig_position[idx] = [landmark.x,landmark.y]
                    fig_text(image,idx,fig_position)
                mouse_position(fig_position,mouse_fig)
                fig_distance=distance(main_fig,fig_position)
                try:
                    print(f"x: {fig_distance[4][0]}, y:{fig_distance[4][0]}, distance:{mt.sqrt(round(fig_distance[4][0],1)**2+round(fig_distance[4][1],1)**2)}")
                except:
                    pass
                    # print(f"x: {fig_distance[4][0]}, y:{fig_distance[4][0]}, distance:error")
                click([[round(mt.sqrt(round(fig_distance[4][0],1)**2+round(fig_distance[4][1],1)**2),2)<=0.1], [round(mt.sqrt(round(fig_distance[4][0],1)**2+round(fig_distance[4][1],1)**2),2)>=0.3]])
                # for idx,i in enumerate(fig_distance):
                    # print(f"{idx} -> x:{i[0]},y:{i[1]}")
                

                # 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 결과 영상 출력
        # cv2.imshow('Hand Tracking', image)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 트레이 아이콘 실행 함수
def setup_tray_icon():
    # 스레드로 백그라운드 작업 실행
    task_thread = threading.Thread(target=background_task)
    task_thread.daemon = True
    task_thread.start()

    # 트레이 아이콘 생성 및 메뉴 설정
    icon = pystray.Icon("test_icon", create_image(), "Test Program")
    icon.menu = pystray.Menu(pystray.MenuItem("Quit", on_quit))  # "Quit" 메뉴 추가

    # 트레이 아이콘 실행
    icon.run()

# 트레이 아이콘 프로그램 시작
if __name__ == "__main__":
    setup_tray_icon()
