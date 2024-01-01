import os

import numpy as np
import scipy.linalg
from numpy.linalg import inv, det
from functions import *
import scipy.special as spsp
import scipy as sp
import scipy.io
from scipy.signal import stft, istft
from scipy.fft import fft, ifft
from f_ssbt import ssbt, issbt, rft, irft

np.set_printoptions(suppress=True, precision=4)


def simulation(freq_mode: str = 'stft', tf_mode: str = 'ctf', res_mode='save'):
    """
    Parameters
    ----------
    freq_mode: str
        Which to use, STFT or SSBT. Defaults to STFT.
    tf_mode: str
        Which to use, CTF or MTF. Defaults to CTF.
    res_mode: str
        Which results to make, 'plot:name_of_metric' or 'save-to-file'. Defaults to save-to-file ('save').
    Returns
    -------

    """

    # Info: Abbreviations
    #       RIR         : Room Impulse Response
    #       SIR         : Signal-to-Interference Ratio
    #       SNR         : Signal-to-Noise Ratio
    #       TF          : Transfer Function
    #       RTF         : Relative Transfer Function
    #       FT          : Frequency Transform
    #       FFT         : Fast Fourier Transform
    #       STFT        : Short-Time Fourier Transform
    #       RFT         : Real Fourier Transform
    #       SSBT        : Single-SideBand Transform
    #       GEFT        : GEneric Frequency Transform


    """
        ------------------
        - Pre-processing -
        ------------------
    """

    freq_mode = freq_mode.lower()
    tf_mode = tf_mode.lower()
    if freq_mode not in ['ssbt', 'stft']:
        raise SyntaxError('Invalid frequency mode.')

    if tf_mode not in ['ctf', 'mtf']:
        raise SyntaxError('Invalid transfer function model.')

    """
        -------------
        - Constants -
        -------------
    """

    # Vars: Constants
    #       dx          : Distance between sensors [m]
    #       c           : Wave speed [m/s]
    #       fs          : Sampling frequency [samples/s]
    #       n_sensors   : Number of sensors in array [scalar]
    #       len_rir     : Length of RIRs [samples]
    #       nperseg     : Number of samples per window [samples]
    #       n_bins      : Number of frequency bins after freq. transform [scalar]
    #       n_bins_star : Number of frequency bins after STFT [scalar]
    #       geft        : GEFT - GEneric Frequency Transform {function}
    #       n_win_rir   : Number of windows in the freq. transform of the RIR [scalar]
    #       sym_freqs   : Simulated frequencies {vector} [Hz]
    #       win_p_fil   : Number of windows to use to calculate each filter [scalar]
    #       dist_fil    : How long to wait to calculate filter again [scalar]
    #       array       : Array of sensors {class Array}
    #       epsilon     : Small value to mitigate divide-by-zero errors [scalar]
    #       SIR_in      : Input SIR [scalar, dB]
    #       SNR_in      : Input SNR [scalar, dB]

    variables = scipy.io.loadmat('io_input/variables.mat')

    dx = variables['delta_x'].item()
    c = variables['c'].item()
    fs = variables['fs'].item()
    n_sensors = variables['n_sensors'].item()
    len_rir = variables['n'].item()
    n_per_seg = 32

    global n_bins, F_lk_star, geft
    match freq_mode:
        case 'ssbt':
            n_bins = n_per_seg
            geft = ssbt
            
        case 'stft':
            n_bins = (1 + n_per_seg)//2 + 1
            geft = stft

    n_bins_star = (1 + n_per_seg)//2 + 1
    n_win_rir = int(np.ceil(2*len_rir/n_per_seg + 1))
    sym_freqs = np.linspace(0, fs, n_per_seg + 1)[:n_bins]
    win_p_fil = 100
    dist_fil = -1  # INFO: If "win_p_fil = -1", uses whole signal
    m_ref = 0
    array = Array((1, n_sensors), 'Array')
    array.init_metrics(sym_freqs.size)  # Initializes metrics

    epsilon = 1e-12
    SIR_in = 0
    SNR_in = 30

    """
        -------------------------
        - Time-domain variables -
        -------------------------
    """
    # Vars: Sources and signals (time)
    #       h_n         : Desired signal's RIR, for each sensor {matrix} [scalar]
    #       g_n         : Undesired signal's RIR, for each sensor {matrix} [scalar]
    #       x_n         : Desired signal at source {vector} [scalar]
    #       v_n         : Undesired signal at source {vector} [scalar]
    #       r_n         : Noise signal, for each sensor {vector} [scalar]
    #       var_x       : Variance of desired signal (at source) [scalar]
    #       var_v       : Variance of undesired signal (at source) [scalar]
    #       var_r       : Variance of noise signal [scalar]
    #       len_x       : Number of samples of x_n [scalar]

    h_n = np.loadtxt('io_input/rir_dx_.csv', delimiter=',')

    len_x = 100000
    x_n = np.random.rand(len_x)         # TODO: load desired signal (speech?)
    var_x = 1

    g_n = np.loadtxt('io_input/rir_v2_.csv', delimiter=',')
    v_n = np.random.rand(2*len_x)       # TODO: load undesired signal (babble?)
    var_v = 1

    r_n = np.random.rand(2*len_x, n_sensors)          # TODO: load/gen noise signal (white?)
    var_r = 1e-4

    # INFO: Array-fixing, so that the desired signal RIR's max. value is at the start of a FT window
    idx_max_h_n = np.where(h_n[m_ref, :] == np.amax(h_n[m_ref, :]))[0][0]
    new_idx_max_h_n = int(np.ceil(idx_max_h_n / (n_per_seg//2)) * (n_per_seg // 2))
    h_n = np.hstack([np.zeros([n_sensors, new_idx_max_h_n - idx_max_h_n]), h_n])
    g_n = np.hstack([np.zeros([n_sensors, new_idx_max_h_n - idx_max_h_n]), g_n])

    l_des_window = new_idx_max_h_n // (n_per_seg // 2)

    len_rir = h_n.shape[1]
    n_win_rir = int(np.ceil(2*len_rir/n_per_seg + 1))
    n_win_R = int(np.ceil(2*r_n.shape[0]/n_per_seg + 1))
    """
        -------------------------------
        - Time-Freq. domain variables -
        -------------------------------
    """

    # Vars: Sources and signals (freq.)
    #       H_lk        : Desired signal's RTF through GEFT, for each sensor {tensor} [scalar]
    #       G_lk        : Undesired signal's RTF through GEFT, for each sensor {tensor} [scalar]
    #       B_lk        : Desired signal's relative RTF through GEFT, for each sensor {tensor} [scalar]
    #       C_lk        : Undesired signal's relative RTF through GEFT, for each sensor {tensor} [scalar]
    #       dx_lk       : Desired signal's relative RTF through GEFT for main window, for each sensor {tensor} [scalar]
    #       X_lk        : Desired signal's FT through GEFT, for each sensor {tensor} [scalar]
    #       V_lk        : Undesired signal's FT through GEFT, for each sensor {tensor} [scalar]
    #       N_lk        : Noise signal's FT through GEFT, for each sensor {tensor} [scalar]
    #       Y_lk        : Observed signal's FT through GEFT, for each sensor {tensor} [scalar]
    #       n_win_X     : Number of windows of X_lk [scalar]
    #       n_win_V     : Number of windows of V_lk [scalar]
    #       n_win_N     : Number of windows of N_lk [scalar]
    #       n_win_Y     : Number of windows of Y_lk [scalar]

    H_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    G_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    N_lk = np.empty((n_bins, n_win_R, n_sensors), dtype=complex)
    for idx in range(n_sensors):
        _, _, Hm_lk = geft(h_n[idx, :], fs, nperseg=n_per_seg)
        H_lk[:, :, idx] = Hm_lk

        _, _, Gm_lk = geft(g_n[idx, :], fs, nperseg=n_per_seg)
        G_lk[:, :, idx] = Gm_lk

        _, _, Rm_lk = geft(r_n[:, idx], fs, nperseg=n_per_seg)
        N_lk[:, :, idx] = Rm_lk

    _, _, X_lk = geft(x_n, fs, nperseg=n_per_seg)
    _, _, V_lk = geft(v_n, fs, nperseg=n_per_seg)

    n_win_X = X_lk.shape[1]
    n_win_V = V_lk.shape[1]
    n_win_R = N_lk.shape[1]
    n_win_Y = n_win_rir+n_win_X-1

    if max(n_win_V, n_win_R) < n_win_X + n_win_rir-1:
        raise AssertionError('Noise signals too short.')

    Y_lk = np.empty((n_bins, n_win_rir+n_win_X-1, n_sensors), dtype=complex)
    for m in range(n_sensors):
        for k_idx in range(n_bins):
            # INFO: CTF convolutions and signal-length correction
            S = np.convolve(H_lk[k_idx, :, m], X_lk[k_idx, :], mode='full')[:n_win_Y]
            I = np.convolve(G_lk[k_idx, :, m], V_lk[k_idx, :], mode='full')[:n_win_Y]
            R = (N_lk[k_idx, :, m])[:n_win_Y]

            # INFO: Variance and SIR/SNR calculations
            S = S / np.sqrt(np.var(S) + epsilon)
            I = I / np.sqrt(np.var(I) + epsilon) / np.sqrt(10 ** (SIR_in / 10))
            R = R / np.sqrt(np.var(R) + epsilon) / np.sqrt(10 ** (SNR_in / 10))
            W = I + R

            # INFO: Calculating desired, undesired, and observed signals
            Y_lk[k_idx, :, m] = S + W

    B_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    C_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    for k_idx in range(n_bins):
        # INFO: Calculating relative TF's
        B_lk[k_idx, :, :] = H_lk[k_idx, :, :] / (H_lk[k_idx, l_des_window, m_ref] + epsilon)
        C_lk[k_idx, :, :] = G_lk[k_idx, :, :] / (G_lk[k_idx, l_des_window, m_ref] + epsilon)

    dx_k = B_lk[:, l_des_window, :]

    # Vars:
    #   "_star" variables are the "true" variable, using STFT instead of GEFT
    #       H_lk_star   : Desired signal's RTF through STFT, for each sensor {tensor} [scalar]
    #       G_lk_star   : Undesired signal's RTF through STFT, for each sensor {tensor} [scalar]
    #       B_lk_star   : Desired signal's relative RTF through STFT, for each sensor {tensor} [scalar]
    #       C_lk_star   : Undesired signal's relative RTF through STFT, for each sensor {tensor} [scalar]
    #       dx_lk_star  : Desired signal's relative RTF through STFT for main window, for each sensor {tensor} [scalar]
    #       X_lk_star   : Desired signal's FT through STFT, for each sensor {tensor} [scalar]
    #       V_lk_star   : Undesired signal's FT through STFT, for each sensor {tensor} [scalar]
    #       N_lk_star   : Noise signal's FT through STFT, for each sensor {tensor} [scalar]
    #       Y_lk_star   : Observed signal's FT through STFT, for each sensor {tensor} [scalar]

    H_lk_star = np.empty((n_bins_star, n_win_rir, n_sensors), dtype=complex)
    G_lk_star = np.empty((n_bins_star, n_win_rir, n_sensors), dtype=complex)
    N_lk_star = np.empty((n_bins_star, n_win_R, n_sensors), dtype=complex)
    for idx in range(n_sensors):
        _, _, Hm_lk_star = stft(h_n[idx, :], fs, nperseg=n_per_seg)
        H_lk_star[:, :, idx] = Hm_lk_star

        _, _, Gm_lk_star = stft(g_n[idx, :], fs, nperseg=n_per_seg)
        G_lk_star[:, :, idx] = Gm_lk_star

        _, _, Rm_lk_star = stft(r_n[:, idx], fs, nperseg=n_per_seg)
        N_lk_star[:, :, idx] = Rm_lk_star

    _, _, X_lk_star = stft(x_n, fs, nperseg=n_per_seg)
    _, _, V_lk_star = stft(v_n, fs, nperseg=n_per_seg)

    Y_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X-1, n_sensors), dtype=complex)
    S_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X-1, n_sensors), dtype=complex)
    I_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X-1, n_sensors), dtype=complex)
    R_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X-1, n_sensors), dtype=complex)
    W_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X-1, n_sensors), dtype=complex)
    for m in range(n_sensors):
        for k_idx in range(n_bins_star):
            # INFO: CTF convolutions and signal-length correction
            S = np.convolve(H_lk_star[k_idx, :, m], X_lk_star[k_idx, :], mode='full')[:n_win_Y]
            I = np.convolve(G_lk_star[k_idx, :, m], V_lk_star[k_idx, :], mode='full')[:n_win_Y]
            R = (N_lk_star[k_idx, :, m])[:n_win_Y]

            # INFO: Variance and SIR/SNR calculations
            S = S / np.sqrt(np.var(S) + epsilon)
            I = I / np.sqrt(np.var(I) + epsilon) / np.sqrt(10**(SIR_in/10))
            R = R / np.sqrt(np.var(R) + epsilon) / np.sqrt(10**(SNR_in/10))
            W = I + R

            # INFO: Calculating desired, undesired, and observed signals
            S_lk_star[k_idx, :, m] = S
            I_lk_star[k_idx, :, m] = I
            R_lk_star[k_idx, :, m] = R
            W_lk_star[k_idx, :, m] = W
            Y_lk_star[k_idx, :, m] = S + W

    B_lk_star = np.empty_like(H_lk_star)
    C_lk_star = np.empty_like(G_lk_star)
    for m in range(n_sensors):
        # INFO: Calculating relative TF's
        B_lk_star[:, :, m] = H_lk_star[:, :, m] / (H_lk_star[l_des_window, :, m] + epsilon)
        C_lk_star[:, :, m] = G_lk_star[:, :, m] / (G_lk_star[l_des_window, :, m] + epsilon)

    dx_k_star = B_lk_star[:, l_des_window, :]

    """
        -------------
        - Filtering -
        -------------
    """

    # Vars: Sources and signals (freq.)
    #       F_lk        : Beamforming filter {tensor} [scalar]
    #       F_lk_star   : Beamforming filter asserted to STFT domain {tensor} [scalar]
    #       n_win_F     : Number of windows of F_lk [scalar]
    #       Corr_Y      : Correlation matrix of Y_lk, for current window and bin {matrix} [scalar]
    #       Sf_lk_star  : Filtered S_lk {matrix} [scalar]
    #       If_lk_star  : Filtered I_lk {matrix} [scalar]
    #       Rf_lk_star  : Filtered R_lk {matrix} [scalar]
    #       Wf_lk_star  : Filtered W_lk {matrix} [scalar]
    #       Yf_lk_star  : Filtered Y_lk {matrix} [scalar] - Z_lk ≡ Yf_lk

    if dist_fil == -1:
        dist_fil = n_win_Y

    # dist_fil = 10
    F_lk = np.empty((n_bins, int(np.ceil(n_win_Y / dist_fil)), n_sensors), dtype=complex)
    n_win_F = F_lk.shape[1]
    for k_idx in range(n_bins):
        dx = dx_k[k_idx, :]
        idx_stt = 0
        idx_end = dist_fil
        l_idx = 0
        while True:
            # INFO: Separating necessary windows of Y_lk, and calculating coherence matrix
            Y = Y_lk[k_idx, idx_stt:idx_end, :]
            Corr_Y = np.empty([n_sensors, n_sensors], dtype=complex)
            for idx_i in range(n_sensors):
                for idx_j in range(idx_i, n_sensors):
                    Corr_Y[idx_i, idx_j] = Y[:, idx_i] @ he(Y[:, idx_j])
                    Corr_Y[idx_j, idx_i] = np.conj(Corr_Y[idx_i, idx_j])
            F_lk[k_idx, l_idx, :] = Corr_Y @ tr(dx) / (np.conj(dx) @ Corr_Y @ tr(dx) + epsilon)

            if idx_end == n_win_Y:
                break
            idx_end += dist_fil
            l_idx += 1
            if idx_end >= win_p_fil:
                idx_stt = idx_end - win_p_fil
            if idx_end >= n_win_Y:
                idx_end = n_win_Y

    match freq_mode:
        case 'stft':
            # Info: F_lk_star.shape = [n_bins_star, n_win_F, n_sensors]
            F_lk_star = F_lk
        case 'ssbt':
            F_lk_star = np.empty((n_bins_star, n_win_F, n_sensors), dtype=float)

            for l_idx in range(n_win_F):
                for m in range(n_sensors):
                    fm_ln = irft(F_lk[:, l_idx, m])
                    Fm_lk = fft(fm_ln)[:Y_lk_star.shape[0]]
                    F_lk_star[:, l_idx, m] = Fm_lk

    Sf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    If_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    Rf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    Wf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    Yf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    
    for k_idx in range(n_bins_star):
        for l_idx in range(n_win_Y):
            F = F_lk_star[k_idx, l_idx//dist_fil, :].reshape(-1, 1)
            S = S_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            I = I_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            R = R_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            W = W_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            Y = Y_lk_star[k_idx, l_idx, :].reshape(-1, 1)

            Sf_lk_star[k_idx, l_idx] = he(F) @ S
            If_lk_star[k_idx, l_idx] = he(F) @ I
            Rf_lk_star[k_idx, l_idx] = he(F) @ R
            Wf_lk_star[k_idx, l_idx] = he(F) @ W
            Yf_lk_star[k_idx, l_idx] = he(F) @ Y

    """
        -----------
        - Metrics -
        -----------
    """

    # Vars: Sources and signals (freq.)
    #       iSINR_lk    : Narrowband input SNR per-window [scalar, dB]
    #       oSINR_lk    : Narrowband output SNR per-window [scalar, dB]
    #       iSINR_lk    : Narrowband input SNR averaged [scalar, dB]
    #       oSINR_lk    : Narrowband output SNR averaged [scalar, dB]
    #       iSINR_lkr   : Narrowband input SNR per-window (w/ rho) [scalar, dB]
    #       oSINR_lkr   : Narrowband output SNR per-window (w/ rho) [scalar, dB]
    #       iSINR_lk    : Narrowband input SNR averaged (w/ rho) [scalar, dB]
    #       oSINR_lk    : Narrowband output SNR averaged (w/ rho) [scalar, dB]

    iSINR_lk = np.empty((n_bins_star, n_win_F), dtype=complex)
    oSINR_lk = np.empty((n_bins_star, n_win_F), dtype=complex)
    iSINR_k = np.empty((n_bins_star,), dtype=complex)
    oSINR_k = np.empty((n_bins_star,), dtype=complex)

    for k_idx in range(n_bins_star):
        idx_stt = 0
        idx_end = dist_fil
        l_idx = 0
        while True:
            S = S_lk_star[k_idx, idx_stt:idx_end, 0].reshape(-1, 1)
            W = W_lk_star[k_idx, idx_stt:idx_end, 0].reshape(-1, 1)

            Sf = Sf_lk_star[k_idx, idx_stt:idx_end, 0].reshape(-1, 1)
            Wf = Wf_lk_star[k_idx, idx_stt:idx_end, 0].reshape(-1, 1)

            var_S = he(S) @ S
            var_W = he(W) @ W

            var_Sf = he(S) @ Sf
            var_Wf = he(W) @ Wf

            iSNR_lk = var_S / (var_W + epsilon)
            oSNR_lk = var_Sf / (var_Wf + epsilon)

            if idx_end == n_win_Y:
                break
            idx_end += dist_fil
            l_idx += 1
            if idx_end >= win_p_fil:
                idx_stt = idx_end - win_p_fil
            if idx_end >= n_win_Y:
                idx_end = n_win_Y

        S = S_lk_star[k_idx, :, 0].reshape(-1, 1)
        W = W_lk_star[k_idx, :, 0].reshape(-1, 1)

        Sf = Sf_lk_star[k_idx, :, 0].reshape(-1, 1)
        Wf = Wf_lk_star[k_idx, :, 0].reshape(-1, 1)

        var_S = he(S) @ S
        var_W = he(W) @ W

        var_Sf = he(S) @ Sf
        var_Wf = he(W) @ Wf

        iSNR_k = var_S / (var_W + epsilon)
        oSNR_k = var_Sf / (var_Wf + epsilon)

    # TODO:
    #       Export measures
    #       Fix missing 'gen_palette'

    input('Stop here')


    F_lk = np.zeros([freqs_t.size, Arr_PA.M], dtype=complex)
    Zk_true = np.zeros([freqs_t_true.size, Arr_PA.M], dtype=complex)
    for k_idx, k in enumerate(freqs_t):
        # # ---------------
        # # Variable reorganization
        Hk = Hlk[k_idx, :, :]
        Gk = Glk[k_idx, :, :]
        dk_x = Hk[0, :]
        Hk_ = Hk[1:, :]

        # # ---------------
        # # Undesired signal correlation matrix
        Corr_w = np.zeros([Arr_PA.M, Arr_PA.M], dtype=complex)
        for idx_l in range(Hk_.shape[0]):
            pH_l = Hk_[idx_l, :].reshape(-1, 1)
            Corr_w += (pH_l @ he(pH_l)) * var_x
        for idx_l in range(Gk.shape[0]):
            pG_l = Gk[idx_l, :]
            Corr_w += (pG_l @ he(pG_l)) * var_v
        Corr_w += np.identity(Arr_PA.M) * var_r
        #        print(Corr_w)
        #        print(det(Corr_w))
        #        input()
        iCorr_w = inv(Corr_w)

        match tf_mode:
            case 'mtf':
                pG_0 = Gk[0, :].reshape(-1, 1)
                Corr_w_tf = (pG_0 @ he(pG_0)) * var_v + np.identity(Arr_PA.M) * var_r
                iCorr_w_tf = inv(Corr_w_tf)
            case 'ctf':
                Corr_w_tf = Corr_w
                iCorr_w_tf = iCorr_w
        # # ---------------
        # # Beamforming
        zk_mvdr = (iCorr_w_tf @ dk_x) / (he(dk_x) @ iCorr_w_tf @ dk_x)
        F_lk[k_idx, :] = zk_mvdr

    for m in range(Arr_PA.M):
        # # ---------------
        # # Frequency-transform correction
        zk_m = F_lk[:, m]
        if freq_mode == 'stft':
            zk_m_true = zk_m
        else:
            zn_m = irft(zk_m)
            zk_m_true = fft(zn_m)[:freqs_t_true.size]
        Zk_true[:, m] = zk_m_true
    Hlk = Hlk_true
    Glk = Glk_true
    F_lk = Zk_true
    freqs_t = freqs_t_true

    for k_idx, k in enumerate(freqs_t):

        # # ---------------
        # # Variable reorganization
        Hk = Hlk[k_idx, :, :]
        Gk = Glk[k_idx, :, :]
        dk_x = Hk[0, :]
        Hk_ = Hk[1:, :]
        zk_mvdr = F_lk[k_idx, :]

        # # ---------------
        # # Undesired signal correlation matrix
        Corr_w = np.zeros([Arr_PA.M, Arr_PA.M], dtype=complex)
        for idx_l in range(Hk_.shape[0]):
            pH_l = Hk_[idx_l, :].reshape(-1, 1)
            Corr_w += (pH_l @ he(pH_l)) * var_x
        for idx_l in range(Gk.shape[0]):
            pG_l = Gk[idx_l, :]
            Corr_w += (pG_l @ he(pG_l)) * var_v
        Corr_w += np.identity(Arr_PA.M) * var_r
        iCorr_w = inv(Corr_w)

        # # ---------------
        # # Metrics
        for a_idx, angle in enumerate(sym_angles):
            d_ta = Tlk_true[k_idx, 0, a_idx, :]
            bpt = np.abs(calcbeam(zk_mvdr, d_ta))
            Arr_PA.beam[k_idx, a_idx] = bpt

        Arr_PA.wng[k_idx] = calc_wng(zk_mvdr, dk_x)
        Arr_PA.gain[k_idx] = calc_gain(zk_mvdr, dk_x, Corr_w, Corr_w[0, 0])
        Arr_PA.gain_expec[k_idx] = np.real(he(dk_x) @ (iCorr_w * Corr_w[0, 0]) @ dk_x)
        Arr_PA.df[k_idx] = calcdf(Arr_PA, zk_mvdr, dk_x, k, c)

        # # ---------------
        # # Result presentation
        freqs = sym_freqs.reshape(-1, ) / 1000
        angles = sym_angles.reshape(-1, )

        beam = dB(vect(Arr_PA.beam).reshape(-1, ))
        wng = dB(Arr_PA.wng.reshape(-1, ))
        gain = dB(Arr_PA.gain.reshape(-1, ))
        gain_expec = dB(Arr_PA.gain_expec.reshape(-1, ))
        df = dB(Arr_PA.df.reshape(-1, ))

    match res_mode:
        case 'plot:gain':
            plt.plot(freqs, gain)
            plt.plot(freqs, gain_expec)
            plt.show()
        case _:
            params = [freqs, angles, wng, df, beam, gain]
            params = fix_decimals(params)
            params = list(params)
            for f_idx, param in enumerate(params):
                params[f_idx] = list(param)
            freqs, angles, wng, df, beam, gain = tuple(params)

            wng_ = list(zip(freqs, wng))
            df_ = list(zip(freqs, df))
            gain_ = list(zip(freqs, gain))
            b_freqs = freqs * len(angles)
            b_angles = []
            for angle_rad in angles:
                b_angles += [angle_rad] * len(freqs)
            beam_ = list(zip(b_freqs, b_angles, beam))

            wng_ = 'freq,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in wng_])
            df_ = 'freq,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in df_])
            beam_ = 'freq,ang,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in beam_])
            gain_ = 'freq,val\n' + '\n'.join([','.join([str(val) for val in item]) for item in gain_])

            tf_mode = tf_mode.upper()
            freq_mode = freq_mode.upper()
            filename = '_' + tf_mode + '_' + freq_mode
            folder = 'io_output/' + tf_mode + '_' + freq_mode + '/'
            if not os.path.isdir('io_output/'):
                os.mkdir('io_output/')
            if not os.path.isdir(folder):
                os.mkdir(folder)
            with open(folder + 'wng' + filename + '.csv', 'w') as f:
                f.write(wng_)
                f.close()
            with open(folder + 'df' + filename + '.csv', 'w') as f:
                f.write(df_)
                f.close()
            with open(folder + 'gain' + filename + '.csv', 'w') as f:
                f.write(gain_)
                f.close()
            with open(folder + 'beam' + filename + '.csv', 'w') as f:
                f.write(beam_)
                f.close()

            beam_min = -20
            beam_max = 0

            # data_defs
            beam_min = r'\def\ymin{{{}}}'.format(beam_min)
            beam_max = r'\def\ymax{{{}}}'.format(beam_max)
            meshcols = r'\def\meshcols{{{}}}'.format(sym_freqs.shape[0])
            meshrows = r'\def\meshrows{{{}}}'.format(len(sym_angles))
            colors = gen_palette(80, 60, 6, 345)
            lightK = r'\definecolor{LightG}{HTML}{3F3F3F}'

            data_defs = [beam_min, beam_max, meshcols, meshrows] + colors + [lightK]
            data_defs = '\n'.join(data_defs)

            with open('io_output/' + 'data_defs.tex', 'w') as f:
                f.write(data_defs)
                f.close()


def main():
    combinations = {
        'A': Params(freq_mode='stft',
                    tf_mode='ctf'),
        'B': Params(freq_mode='stft',
                    tf_mode='mtf'),
        'C': Params(freq_mode='ssbt',
                    tf_mode='ctf'),
        'D': Params(freq_mode='ssbt',
                    tf_mode='mtf'),
    }

    for comb in combinations.keys():
        modes = combinations[comb]
        simulation(freq_mode=modes.freq_mode,
                   tf_mode=modes.tf_mode,
                   res_mode='save')


if __name__ == '__main__':
    main()
