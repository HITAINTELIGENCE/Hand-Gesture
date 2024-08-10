import cv2
import streamlit as st
import mediapipe as mp
import numpy as np 
import torch 
import models.DDNet_Original as Net  
import cv2
import numpy as np
import time
import torch
import mediapipe as mp
from tqdm import tqdm
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
from scipy.ndimage import zoom
import random
import requests
import json

class Config():
    def __init__(self):
        self.frame_l = 32 # the length of frames
        self.joint_n = 22 # the number of joints
        self.joint_d = 3 # the dimension of joints
        self.clc_num = 14 # the number of class
        self.feat_d = 231
        self.filters = 64
def pad_arrays(data):
    max_length = max(max(arr.shape[0] for arr in frame) for frame in data)
    return np.array([[np.pad(arr, (0, max_length - arr.shape[0]), 'constant') for arr in frame] for frame in data])
def zoom(p, target_l, joints_num, joints_dim):
    l = p.shape[0]
    p_new = np.empty(shape=[target_l, joints_num, joints_dim])
    print(p.shape)
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n], 3)
            p_new[:,m,n] = inter.zoom(p[:,m,n], target_l/l)[:target_l]
    return p_new
def sampling_frame(p, C):
    full_l = p.shape[0] # full length
    if random.uniform(0, 1) < 0.5: # alignment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        s = random.randint(0, full_l - int(valid_l))
        e = s + valid_l # sample end point
        p = p[int(s):int(e), :, :]
    else: # without alignment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        index = np.sort(np.random.choice(range(0, full_l), int(valid_l), replace=False))
        p = p[index, :, :]
    p = zoom(p, C.frame_l, C.joint_n, C.joint_d)
    return p
def norm_train(p):
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    p[:, :, 2] = p[:, :, 2] - np.mean(p[:, :, 2])
    return p

def norm_train2d(p):
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    return p
def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        # distance max
        d_m = cdist(p[f], np.concatenate([p[f], np.zeros([1, C.joint_d])]), 'euclidean')
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    return M
def data_generator_rt(T, C):
    X_0 = []
    X_1 = []
    print(T.shape)
    T = np.expand_dims(T, axis = 0)
    for i in tqdm(range(len(T))):
        p = np.copy(T[i])
        p = zoom(p, target_l=C.frame_l, joints_num=C.joint_n, joints_dim=C.joint_d)

        M = get_CG(p, C)

        X_0.append(M)
        p = norm_train2d(p)

        X_1.append(p)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)

    return X_0, X_1
def main():
    st.title("Webcam Live Feed with Hand Tracking")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    C = Config()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    time0 = 0
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.6
    camera = cv2.VideoCapture(0)
    predictions = []
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
                print(type(X_test_rt_1), type(X_test_rt_2))
                try:
                    response = requests.post("http://127.0.0.1:8000/predict", json={
                        "X_test_rt_1": X_test_rt_1_list,
                        "X_test_rt_2": X_test_rt_2_list
                    })

                    if response.status_code == 200:
                        prediction = response.json().get("prediction", "Unknown")
                        predictions.append(prediction)
                        st.write(f"Prediction: {predictions[-1]}")
                    else:
                        st.write("Error:", response.text)
                except Exception as e:
                    st.write(f"Request failed: {str(e)}")
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

    camera.release()

if __name__ == "__main__":
    main()
