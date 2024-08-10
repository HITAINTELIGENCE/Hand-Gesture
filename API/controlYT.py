import streamlit as st
import cv2
import mediapipe as mp

# Thiết lập MediaPipe
mp_hands = mp.solutions.hands

# Hàm xử lý và nhận diện cử chỉ tay (giả định vẫy tay)
def detect_gesture(frame):
    # Logic nhận diện cử chỉ vẫy tay
    # Đây chỉ là ví dụ, bạn cần cập nhật logic phù hợp
    gesture = "Wave"  # Giả định nhận diện được cử chỉ vẫy tay
    return gesture

# URL video YouTube
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# UI trong Streamlit
st.title("YouTube Control with Hand Gestures")
st.write("Detecting hand gestures to control YouTube")

# Hiển thị video YouTube
video_placeholder = st.empty()

# Mở webcam
cap = cv2.VideoCapture(0)

playing = False  # Trạng thái video

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý khung hình
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Nhận diện cử chỉ
        gesture = detect_gesture(frame)
        
        if gesture == "Wave":
            if playing:
                st.write("Pausing video...")
                video_placeholder.empty()  # Tạm dừng video bằng cách xóa placeholder
                playing = False
            else:
                st.write("Playing video...")
                video_placeholder.video(video_url)  # Phát video
                playing = True

cap.release()
