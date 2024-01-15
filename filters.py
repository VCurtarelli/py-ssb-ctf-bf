import numpy as np
import functions as f
from numpy.linalg import inv


def stft_mvdr_1(signal, n_sensors, D, epsilon):
    Corr_sig = np.empty((n_sensors, n_sensors), dtype=complex)
    for idx_i in range(n_sensors):
        sig_i = signal[:, idx_i].reshape(-1, 1)
        for idx_j in range(n_sensors):
            sig_j = signal[:, idx_j].reshape(-1, 1)
            corr = (f.he(sig_i) @ sig_j).item()
            Corr_sig[idx_i, idx_j] = corr
            Corr_sig[idx_j, idx_i] = np.conj(corr)
    try:
        iCorr_sig = inv(Corr_sig)
        flk = ((iCorr_sig @ D) / (f.he(D) @ iCorr_sig @ D + epsilon)).reshape(n_sensors)
    except np.linalg.LinAlgError:
        flk = 0
        
    return flk


def stft_mvdr_2(signal, n_sensors, B, l_des_win, epsilon):
    Corr_sig = np.empty((n_sensors, n_sensors), dtype=complex)
    iD = np.zeros((B.shape[0], 1))
    iD[l_des_win] = 1
    B = f.tr(B)
    for idx_i in range(n_sensors):
        sig_i = signal[:, idx_i].reshape(-1, 1)
        for idx_j in range(n_sensors):
            sig_j = signal[:, idx_j].reshape(-1, 1)
            corr = (f.he(sig_i) @ sig_j).item()
            Corr_sig[idx_i, idx_j] = corr
            Corr_sig[idx_j, idx_i] = np.conj(corr)
    try:
        iCorr_sig = inv(Corr_sig)
        flk = ((iCorr_sig @ B) @ inv(f.he(B) @ iCorr_sig @ B + epsilon) @ iD).reshape(n_sensors)
    except np.linalg.LinAlgError:
        flk = 0
    
    return flk


def nssbt_mvdr_1(signal, n_sensors, D, epsilon):
    return stft_mvdr_1(signal, n_sensors, D, epsilon)


def nssbt_mvdr_2(signal, n_sensors, B, l_des_win, epsilon):
    return stft_mvdr_2(signal, n_sensors, B, l_des_win, epsilon)


def tssbt_mvdr_1(signal1, signal2, n_sensors, arr_delay, D, epsilon):
    if (signal1 == signal2).all():
        n_sensors_ = 1 * n_sensors
        signal = signal1
    else:
        n_sensors_ = 2 * n_sensors
        signal = np.hstack([signal1, signal2])
        D = f.he(arr_delay) @ D
        
    Corr_sig = np.empty((n_sensors_, n_sensors_), dtype=float)
    idd = np.array([[1], [0]])
    for idx_i in range(n_sensors_):
        sig_i = signal[:, idx_i].reshape(-1, 1)
        for idx_j in range(n_sensors_):
            sig_j = signal[:, idx_j].reshape(-1, 1)
            corr = np.real(f.tr(sig_i) @ sig_j).item()
            Corr_sig[idx_i, idx_j] = corr
            Corr_sig[idx_j, idx_i] = corr
            
    Q = np.hstack([np.real(D), np.imag(D)])
    try:
        iCorr_sig = inv(Corr_sig + np.eye(n_sensors_) * epsilon)
        flk = (iCorr_sig @ Q @ inv(f.tr(Q) @ iCorr_sig @ Q) @ idd).reshape(n_sensors_)
    except np.linalg.LinAlgError:
        flk = np.zeros(n_sensors_)
        
    return flk
                

def tssbt_mvdr_2(signal1, signal2, n_sensors, arr_delay, B, l_des_win, epsilon):
    if (signal1 == signal2).all():
        n_sensors_ = 1 * n_sensors
        signal = signal1
    else:
        n_sensors_ = 2 * n_sensors
        signal = np.hstack([signal1, signal2])
        B = f.tr(B)
        B = f.he(arr_delay) @ B
    
    Corr_sig = np.empty((n_sensors_, n_sensors_), dtype=float)
    Q = np.hstack([np.real(B), np.imag(B)])
    iD = np.zeros((Q.shape[0], 1))
    iD[l_des_win] = 1
    for idx_i in range(n_sensors_):
        sig_i = signal[:, idx_i].reshape(-1, 1)
        for idx_j in range(n_sensors_):
            sig_j = signal[:, idx_j].reshape(-1, 1)
            corr = np.real(f.tr(sig_i) @ sig_j).item()
            Corr_sig[idx_i, idx_j] = corr
            Corr_sig[idx_j, idx_i] = corr
            
    try:
        iCorr_sig = inv(Corr_sig)
        flk = (iCorr_sig @ Q @ inv(f.tr(Q) @ iCorr_sig @ Q) @ iD).reshape(n_sensors_)
    except np.linalg.LinAlgError:
        flk = np.zeros(n_sensors_)
        
    return flk
