from models.DDNet_Original import DDNet_Original as Net

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

#r"D:\HIT_PRODUCT_2024\HandGesture\DD-Net-Pytorch\exper   iments\1720754240\model.pt"
# D:\HIT_PRODUCT_2024\HandGesture\DD-Net-Pytorch\experiments\1721637501
#r"D:\HIT_PRODUCT_2024\HandGesture\DD-Net-Pytorch\experiments\1721713458\model.pt"

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

C = Config()
def data_generator_rt(T,C):
    X_0 = []
    X_1 = []
    print(T.shape)
    T = np.expand_dims(T, axis = 0)
    for i in tqdm(range(len(T))):
        p = np.copy(T[i])
        p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)

        M = get_CG(p,C)

        X_0.append(M)
        p = norm_train2d(p)

        X_1.append(p)

    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)

    return X_0,X_1

def zoom(p,target_l,joints_num,joints_dim):
    l = p.shape[0]
    p_new = np.empty(shape = [target_l,joints_num,joints_dim])
    print(p.shape)
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]
    return p_new

def sampling_frame(p,C):
    full_l = p.shape[0] # full length
    if random.uniform(0,1)<0.5: # aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        s = random.randint(0, full_l-int(valid_l))
        e = s+valid_l # sample end point
        p = p[int(s):int(e),:,:]
    else: # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9,1)*full_l)
        index = np.sort(np.random.choice(range(0,full_l),int(valid_l),replace=False))
        p = p[index,:,:]
    p = zoom(p,C.frame_l,C.joint_n,C.joint_d)
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
    # normolize to start point, use the center for hand case
    # p[:,:,0] = p[:,:,0]-p[:,3:4,0]
    # p[:,:,1] = p[:,:,1]-p[:,3:4,1]
    # p[:,:,2] = p[:,:,2]-p[:,3:4,2]
    # # return p

    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    p[:, :, 2] = p[:, :, 2] - np.mean(p[:, :, 2])
    return p


def norm_train2d(p):
    p[:, :, 0] = p[:, :, 0] - np.mean(p[:, :, 0])
    p[:, :, 1] = p[:, :, 1] - np.mean(p[:, :, 1])
    # p[:,:,2] = p[:,:,2]-np.mean(p[:,:,2])
    return p
def main():
    # labels = ['xin chao rat vui duoc gap ban', 'tam biet hen gap lai', 'xin cam on ban that tot bung',
    #           'toi xin loi ban co sao khong', 'toi yeu gia dinh va ban be', 'toi la hoc sinh', 'toi thich dong vat',
    #           'toi an com', 'toi song o viet nam', 'toi la nguoi diec']
    labels = ['Tap','Grab','RC', 'Pinch', 'Expand', 'RCC', 'SR', 'SL', 'SD','SU' , 'SV', 'S+','SX', 'Shake']


    colors = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),(255, 255, 255),(255, 255, 255),(255, 255, 255), (255, 255, 255), (255, 255, 255)]

    # define mediapipe solutions
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hand = mp.solutions.hands

    time0 = 0
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.6
    Y_pred = np.array([])
    # access webcam with opencv
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.9, max_num_hands=1) as pose:
        while cap.isOpened():
            success, image = cap.read()
            heigh, width, _ = image.shape
            # Flip the image horizontally for a selfie-view display.
            # image = cv2.flip(image, 1)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hand.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_pose_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            #     print(results.multi_hand_landmarks)
            #     if results.multi_hand_landmarks:
                    keypoint = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                    keypoint = np.array_split(keypoint, 22)
                    # print(keypoint)
                    sequence.append(keypoint)
                    # print(sequence)
                    sequence = sequence[-32:]
                    # print(sequence)

                if len(sequence) == 32:
                    sequence = np.array(sequence, dtype=object)
                    sequence = pad_arrays(sequence)
                    X_test_rt_1, X_test_rt_2 = data_generator_rt(sequence[-32:], C)
                    X_test_rt_1 = torch.from_numpy(X_test_rt_1).type('torch.FloatTensor')
                    X_test_rt_2 = torch.from_numpy(X_test_rt_2).type('torch.FloatTensor')

                    device = torch.device('cpu')
                    models = Net(frame_l=32, joint_n=22, joint_d=3, class_num=14, feat_d=231, filters=64)
                    models.load_state_dict(
                        torch.load(r"model.pt",
                                   weights_only=True), strict=False)
                    # model = Net.to(device)
                    models.eval()
                    with torch.no_grad():
                        Y_pred = models(X_test_rt_1, X_test_rt_2).cpu().numpy()
                    models.eval()
                    print(Y_pred)
                    if np.max(Y_pred) >= threshold:
                        sentence.append(labels[np.argmax(Y_pred)])
                    else:
                        sentence.append(None)
                    # Viz probabilities

                    sequence = []

                    print(sentence)
            # Show fps
            time1 = time.time()
            fps = 1 / (time1 - time0)
            time0 = time1
            cv2.putText(image, 'FPS:' + str(int(fps)), (3, 475), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            #         cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, ''.join(str(sentence[-1:])), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            cv2.imshow('SIGNTEGRATE', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if "__name__" == main():
    main()