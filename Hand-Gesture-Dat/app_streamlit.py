import os
import streamlit as st
import models.DDNet_Original as Net  
import cv2
import numpy as np
import torch
import mediapipe as mp
from tqdm import tqdm
from scipy.ndimage import zoom, interpolation as inter
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
import random
import requests
from pathlib import Path
from spire.presentation import Presentation

# Cấu hình các tham số
class Config:
    def __init__(self):
        self.frame_l = 32  # chiều dài của khung hình
        self.joint_n = 22  # số lượng khớp
        self.joint_d = 3   # số chiều của khớp
        self.clc_num = 14  # số lớp
        self.feat_d = 231
        self.filters = 64

# Hàm để đệm các mảng
def pad_arrays(data):
    max_length = max(max(arr.shape[0] for arr in frame) for frame in data)
    return np.array([[np.pad(arr, (0, max_length - arr.shape[0]), 'constant') for arr in frame] for frame in data])

# Hàm phóng đại mảng
def zoom_array(p, target_l, joints_num, joints_dim):
    l = p.shape[0]
    p_new = np.empty(shape=[target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p[:, m, n], target_l / l)[:target_l]
    return p_new

# Hàm lấy mẫu khung hình
def sampling_frame(p, C):
    full_l = p.shape[0]  # chiều dài đầy đủ
    if random.uniform(0, 1) < 0.5:  # lấy mẫu căn chỉnh
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        s = random.randint(0, full_l - int(valid_l))
        e = s + valid_l  # điểm cuối mẫu
        p = p[int(s):int(e), :, :]
    else:  # lấy mẫu không căn chỉnh
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        index = np.sort(np.random.choice(range(0, full_l), int(valid_l), replace=False))
        p = p[index, :, :]
    p = zoom_array(p, C.frame_l, C.joint_n, C.joint_d)
    return p

# Hàm chuẩn hóa dữ liệu 3D
def norm_train(p):
    p[:, :, 0] -= np.mean(p[:, :, 0])
    p[:, :, 1] -= np.mean(p[:, :, 1])
    p[:, :, 2] -= np.mean(p[:, :, 2])
    return p

# Hàm chuẩn hóa dữ liệu 2D
def norm_train2d(p):
    p[:, :, 0] -= np.mean(p[:, :, 0])
    p[:, :, 1] -= np.mean(p[:, :, 1])
    return p

# Hàm lấy ma trận liên kết khớp
def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        d_m = cdist(p[f], np.concatenate([p[f], np.zeros([1, C.joint_d])]), 'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M

# Hàm sinh dữ liệu thời gian thực
def data_generator_rt(T, C):
    X_0, X_1 = [], []
    T = np.expand_dims(T, axis=0)
    for i in tqdm(range(len(T))):
        p = np.copy(T[i])
        p = zoom_array(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)
        M = get_CG(p, C)
        X_0.append(M)
        p = norm_train2d(p)
        X_1.append(p)
    X_0, X_1 = np.stack(X_0), np.stack(X_1)
    return X_0, X_1

# Hàm hiển thị slide
def print_slide(camera_placeholder, folderPath, pathImages, imgNumber):
    imgCurrent = cv2.imread(os.path.join(folderPath, pathImages[imgNumber]))
    camera_placeholder.image(imgCurrent, caption='Slide Image', use_column_width=True)

# Hàm xử lý file PowerPoint và chuyển đổi các slide thành ảnh
def process_pptx(file_path, output_dir):
    for file in output_dir.iterdir():
        file.unlink()
    presentation = Presentation()
    presentation.LoadFromFile(file_path)
    for i, slide in enumerate(presentation.Slides):
        file_name = output_dir / f"image{i}.png"
        image = slide.SaveAsImage()
        image.Save(str(file_name))
        image.Dispose()
    presentation.Dispose()

def main():
    C = Config()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    sequence, predictions = [], []
    threshold, delay, counter, imgNumber = 0.6, 30, 0, 0
    buttonPressed = False
    folderPath = 'slides_images'

    col1, col2 = st.columns([1, 1/4])
    with col1:
        camera_placeholder = st.empty()

    st.title("Webcam Live Feed with Hand Tracking")

    output_dir = Path(folderPath)
    if not output_dir.exists():
        output_dir.mkdir()
    run = st.checkbox('Run')

    with col2:
        camera = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

    uploaded_file = st.file_uploader("Upload PowerPoint file", type=["pptx"])

    if uploaded_file is not None:
        pptx_path = "uploaded_pptx.pptx"
        with open(pptx_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("Processing your file...")
        process_pptx(pptx_path, output_dir)

    while run:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoint = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                keypoint = np.array_split(keypoint, 22)
                sequence.append(keypoint)
                sequence = sequence[-32:]

            if len(sequence) == 32:
                sequence_np = np.array(sequence, dtype=object)
                sequence_np = pad_arrays(sequence_np)
                X_test_rt_1, X_test_rt_2 = data_generator_rt(sequence_np[-32:], C)

                X_test_rt_1 = torch.from_numpy(X_test_rt_1).type(torch.FloatTensor)
                X_test_rt_2 = torch.from_numpy(X_test_rt_2).type(torch.FloatTensor)
                X_test_rt_1_list = X_test_rt_1.tolist()
                X_test_rt_2_list = X_test_rt_2.tolist()
                
                try:
                    response = requests.post("http://127.0.0.1:8000/predict", json={
                        "X_test_rt_1": X_test_rt_1_list,
                        "X_test_rt_2": X_test_rt_2_list
                    })

                    if response.status_code == 200:
                        prediction = response.json().get("prediction", "Unknown")
                        predictions.append(prediction)
                        last_three_predictions = predictions[-3:]
                        most_common_prediction = max(set(last_three_predictions), key=last_three_predictions.count)
                        st.write(f"Most common prediction in last 3: {most_common_prediction}")

                        pathImages = sorted(os.listdir(folderPath), key=len)
                        print_slide(camera_placeholder, folderPath, pathImages, imgNumber)

                        if most_common_prediction == 'SL' and not buttonPressed:
                            print("Left")
                            buttonPressed = True
                            if imgNumber > 0:
                                imgNumber -= 1

                        if most_common_prediction == 'SR' and not buttonPressed:
                            print("Right")
                            buttonPressed = True
                            if imgNumber < len(pathImages) - 1:
                                imgNumber += 1
                        
                        print_slide(camera_placeholder, folderPath, pathImages, imgNumber)

                        if buttonPressed:
                            counter += 1
                            if counter > delay:
                                counter = 0
                                buttonPressed = False
                    else:
                        st.write("Error:", response.text)
                except Exception as e:
                    st.write(f"Request failed: {str(e)}")
        
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

    camera.release()

if __name__ == "__main__":
    main()
