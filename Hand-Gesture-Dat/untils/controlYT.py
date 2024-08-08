import cv2
import mediapipe as mp
import pyautogui

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Hàm để điều khiển YouTube
def control_youtube(label):
    if label == "play_pause":
        pyautogui.press('space')  # Phím tắt để phát/tạm dừng video
    else:
        print("Label không hợp lệ!")

# Hàm để nhận diện cử chỉ tay và tạo nhãn
def get_gesture_label(landmarks):
    # Kiểm tra nếu tất cả 5 ngón tay đều được giơ lên
    fingers_up = [landmarks[8].y < landmarks[6].y,  # Ngón trỏ
                  landmarks[12].y < landmarks[10].y,  # Ngón giữa
                  landmarks[16].y < landmarks[14].y,  # Ngón áp út
                  landmarks[20].y < landmarks[18].y,  # Ngón út
                  landmarks[4].x < landmarks[3].x]  # Ngón cái giơ lên khi nó ở bên trái ngón cái gốc (chỉ cho tay phải)

    if all(fingers_up):
        return "play_pause"
    return None

# Hàm để tính toán và vẽ bounding box
def draw_bounding_box(frame, hand_landmarks):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    # Vẽ bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Khởi động camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi hình ảnh sang RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Nhận diện bàn tay
    results = hands.process(image)

    # Nếu phát hiện bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ landmarks trên bàn tay
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Tạo nhãn từ landmarks
            label = get_gesture_label(hand_landmarks.landmark)
            if label:
                control_youtube(label)

            # Vẽ bounding box xung quanh bàn tay
            draw_bounding_box(frame, hand_landmarks.landmark)

    # Hiển thị khung hình
    cv2.imshow('Hand Gesture Recognition', frame)

    # Nhấn 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng tất cả các cửa sổ
cap.release()
cv2.destroyAllWindows()
hands.close()
