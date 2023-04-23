import pyautogui

screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
print(screenWidth, screenHeight)

currentMouseX, currentMouseY = pyautogui.position()

pyautogui.moveTo(100, 150)