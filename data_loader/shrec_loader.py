#! /usr/bin/env python
#! coding:utf-8:w

import numpy as np
from sklearn import preprocessing
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
from utils import *  # noqa
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
current_file_dirpath = Path(__file__).parent.absolute()
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt

def load_shrec_data(
        train_path=current_file_dirpath / Path("../data/SHREC/train.pkl"),
        test_path=current_file_dirpath / Path("../data/SHREC/test.pkl"),
):
    Train = pickle.load(open(train_path, "rb"))
    Test = pickle.load(open(test_path, "rb"))
    print("Loading SHREC Dataset")
    dummy = None  # return a dummy to provide a similar interface with JHMDB one
    return Train, Test, None

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

class SConfig():
    def __init__(self):
        self.frame_l = 32  # the length of frames
        self.joint_n = 22  # the number of joints
        self.joint_d = 3  # the dimension of joints
        self.class_coarse_num = 14
        self.class_fine_num = 28
        self.feat_d = 231
        self.filters = 64


class Sdata_generator:
    def __init__(self, label_level='coarse_label'):
        self.label_level = label_level

    # le is None to provide a unified interface with JHMDB datagenerator
    def __call__(self, T, C, le=None):
        X_0 = []
        X_1 = []
        Y = []
        for i in tqdm(range(len(T['pose']))):
            p = np.copy(T['pose'][i].reshape([-1, 22, 3]))
            # p.shape (frame,joint_num,joint_coords_dims)
            p = zoom(p, target_l=C.frame_l,
                     joints_num=C.joint_n, joints_dim=C.joint_d)
            # p.shape (target_frame,joint_num,joint_coords_dims)
            # label = np.zeros(C.clc_num)
            # label[labels[i]] = 1
            label = (T[self.label_level])[i] - 1
            # M.shape (target_frame,(joint_num - 1) * joint_num / 2)
            M = get_CG(p, C)

            X_0.append(M)
            X_1.append(p)
            Y.append(label)

        self.X_0 = np.stack(X_0)
        self.X_1 = np.stack(X_1)
        self.Y = np.stack(Y)
        return self.X_0, self.X_1, self.Y
def zoom(p,target_l=32,joints_num=20,joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l,joints_num,joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]
    return p_new

if __name__ == '__main__':
    Train, Test, le = load_shrec_data()
    print(len(Test['pose'][0][0]))
    C = SConfig()
    X_0_t, X_1_t, Y_t = Sdata_generator('coarse_label')(Test, C, 'coarse_label')
    print(X_0_t)
    X_0, X_1, Y = Sdata_generator('fine_label')(Train, C, 'fine_label')
    print(Y)
    print("X_0.shape", X_0.shape)
    print("X_1.shape", X_1.shape)
    print("X_0_t.shape", X_0_t.shape)
    print("X_1_t.shape", X_1_t.shape)
