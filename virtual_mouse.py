import cv2
import mediapipe as mp
import pyautogui
import math

# Webcam setup
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smoothing variables
prev_x, prev_y = 0, 0
smoothening = 5

# Click delay control
click_delay = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape

    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []

            # Get landmark positions
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Important points
            x1, y1 = lm_list[8][1], lm_list[8][2]   # Index
            x2, y2 = lm_list[4][1], lm_list[4][2]   # Thumb
            x3, y3 = lm_list[12][1], lm_list[12][2] # Middle

            # Draw tracking point
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)

            # Convert to screen coordinates
            mouse_x = screen_w * (x1 / w)
            mouse_y = screen_h * (y1 / h)

            # Smooth movement
            curr_x = prev_x + (mouse_x - prev_x) / smoothening
            curr_y = prev_y + (mouse_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distance calculations
            dist_click = math.hypot(x2 - x1, y2 - y1)   # Thumb-Index
            dist_right = math.hypot(x2 - x3, y2 - y3)   # Thumb-Middle
            dist_scroll = abs(y1 - y3)                  # Index-Middle vertical

            # Left Click
            if dist_click < 25 and click_delay == 0:
                pyautogui.click()
                click_delay = 10

            # Right Click
            if dist_right < 25 and click_delay == 0:
                pyautogui.rightClick()
                click_delay = 10

            # Scroll
            if dist_scroll < 40:
                pyautogui.scroll(20)

            # Reduce delay
            if click_delay > 0:
                click_delay -= 1

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Display
    cv2.imshow("Virtual Mouse", img)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()