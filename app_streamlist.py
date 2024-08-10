import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import requests
import torch
from tqdm import tqdm
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt
from scipy.spatial.distance import cdist
from scipy.ndimage import zoom
import random

class Config():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.clc_num = 14  # the number of class
        self.feat_d = 231
        self.filters = 64

def pad_arrays(data):
    max_length = max(max(arr.shape[0] for arr in frame) for frame in data)
    return np.array([[np.pad(arr, (0, max_length - arr.shape[0]), 'constant') for arr in frame] for frame in data])

C = Config()

def data_generator_rt(T, C):
    X_0 = []
    X_1 = []
    T = np.expand_dims(T, axis=0)
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

def zoom(p, target_l, joints_num, joints_dim):
    l = p.shape[0]
    p_new = np.empty(shape=[target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            p_new[:, m, n] = inter.zoom(p[:, m, n], target_l/l)[:target_l]
    return p_new

def sampling_frame(p, C):
    full_l = p.shape[0]  # full length
    if random.uniform(0, 1) < 0.5:  # alignment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        s = random.randint(0, full_l - int(valid_l))
        e = s + valid_l  # sample end point
        p = p[int(s):int(e), :, :]
    else:  # without alignment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        index = np.sort(np.random.choice(range(0, full_l), int(valid_l), replace=False))
        p = p[index, :, :]
    p = zoom(p, C.frame_l, C.joint_n, C.joint_d)
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

def norm_train(p):
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    p[:, :, 2] = p[:, :, 2] - np.mean(p[:, :, 2])
    return p

def norm_train2d(p):
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    return p

# Streamlit app
st.title('Hand Gesture Recognition with MediaPipe')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands

# Create a Streamlit button for starting and stopping
start_button = st.button('Start')
stop_button = st.button('Stop')

if 'is_running' not in st.session_state:
    st.session_state.is_running = False

if start_button:
    st.session_state.is_running = True

if stop_button:
    st.session_state.is_running = False

# Create a Streamlit placeholder for video
video_placeholder = st.empty()

# Initialize VideoCapture
cap = cv2.VideoCapture(0)
sequence = []

with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while True:
        if st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    keypoints = np.array_split(keypoints, 22)
                    sequence.append(keypoints)
                    sequence = sequence[-32:]

                if len(sequence) == 32:
                    sequence = np.array(sequence, dtype=object)
                    sequence = pad_arrays(sequence)
                    X_test_rt_1, X_test_rt_2 = data_generator_rt(sequence[-32:], C)

                    data = {
                        'X_test_rt_1': X_test_rt_1.tolist(),
                        'X_test_rt_2': X_test_rt_2.tolist()
                    }

                    try:
                    response = requests.post('http://127.0.0.1:8000/predict', json=data)
                    response.raise_for_status()  # Raises HTTPError for bad responses
                    prediction = response.json()
                    st.write('Prediction:', prediction)
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"HTTP error occurred: {http_err}")
                except requests.exceptions.RequestException as req_err:
                    st.error(f"Error in API request: {req_err}")
                except Exception as err:
                    st.error(f"An error occurred: {err}")


                    sequence = []

            # Clear the placeholder and show the frame
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        else:
            video_placeholder.empty()

cap.release()
cv2.destroyAllWindows()
