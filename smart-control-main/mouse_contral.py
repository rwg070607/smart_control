import pyautogui
import time

# 잠시 대기 (5초 후에 마우스 이동)
time.sleep(5)

# 마우스를 (100, 100) 위치로 이동
pyautogui.moveTo(100, 100, duration=1)  # duration은 이동하는데 걸리는 시간

# 마우스 클릭
pyautogui.click()

# 오른쪽 클릭
pyautogui.rightClick()

# 더블 클릭
pyautogui.doubleClick()

# 드래그 (지정한 위치로 드래그)
pyautogui.dragTo(200, 200, duration=1)

# 현재 마우스 위치 출력
print(pyautogui.position())
