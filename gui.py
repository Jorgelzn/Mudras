import pyautogui
import time
screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
print(screenWidth, screenHeight)

currentMouseX, currentMouseY = pyautogui.position()

pyautogui.moveTo(100, 150)

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)

print('down')
pyautogui.keyDown('w') 
time.sleep(3)
print('up')
pyautogui.keyUp('w') 