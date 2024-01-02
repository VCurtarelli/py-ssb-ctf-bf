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
import gen_palette as gp
import soundfile as sf
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)


def simulation(freq_mode: str = 'stft', sig_mode='random'):
    """
    Parameters
    ----------
    freq_mode: str
        Which to use, STFT or SSBT. Defaults to STFT.
    sig_mode: str
        Which signals to use, 'random' or 'load'. Defaults to 'random'.
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
    if freq_mode not in ['ssbt', 'stft']:
        raise SyntaxError('Invalid frequency mode.')

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
    n_per_seg = 64

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
    sym_freqs = np.linspace(0, fs, n_per_seg + 1)[:n_bins_star]
    dist_fil = 25
    win_p_fil = 250  # INFO: If "win_p_fil = -1", uses whole signal
    m_ref = 0
    array = Array((1, n_sensors), 'Array')
    array.init_metrics(sym_freqs.size)  # Initializes metrics

    epsilon = 1e-15
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
    #       len_x       : Number of samples of x_n [scalar]

    h_n = np.loadtxt('io_input/rir_dx_.csv', delimiter=',')
    g_n = np.loadtxt('io_input/rir_v2_.csv', delimiter=',')

    match sig_mode:
        case 'load':
            x_n, samplerate = sf.read('io_input/audio_speech_female.flac')
            x_n = resample(x_n, samplerate, fs)
            v_n, samplerate = sf.read('io_input/audio_music_abba.flac')
            v_n = resample(v_n, samplerate, fs)
            r_n, samplerate = sf.read('io_input/audio_noise_wgn.flac')
            r_n = resample(r_n, samplerate, fs).reshape(-1, 1)
            nr_n = []
            for m in range(n_sensors):
                nr_n.append(np.roll(r_n, int(m/n_sensors * r_n.size)))
            r_n = np.hstack(nr_n)
        case 'random', _:
            len_x = 100000
            x_n = np.random.rand(len_x)         # TODO: load desired signal (speech?)

            g_n = np.loadtxt('io_input/rir_v2_.csv', delimiter=',')
            v_n = np.random.rand(2*len_x)       # TODO: load undesired signal (babble?)

            r_n = np.random.rand(2*len_x, n_sensors)          # TODO: load/gen noise signal (white?)

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
        win_p_fil = n_win_Y

    F_lk = np.empty((n_bins, int(np.ceil(n_win_Y / dist_fil)), n_sensors), dtype=complex)
    n_win_F = F_lk.shape[1]
    for k_idx in range(n_bins):
        dx = dx_k[k_idx, :]
        for l_idx in range(n_win_F):
            # INFO: Separating necessary windows of Y_lk, and calculating coherence matrix
            idx_stt = max(0, (l_idx+1)*dist_fil-win_p_fil)
            idx_end = min((l_idx+1)*dist_fil, n_win_Y)
            Y = Y_lk[k_idx, idx_stt:idx_end, :]
            Corr_Y = np.empty([n_sensors, n_sensors], dtype=complex)
            for idx_i in range(n_sensors):
                for idx_j in range(idx_i, n_sensors):
                    Corr_Y[idx_i, idx_j] = Y[:, idx_i] @ he(Y[:, idx_j])
                    Corr_Y[idx_j, idx_i] = np.conj(Corr_Y[idx_i, idx_j])
            F_lk[k_idx, l_idx, :] = (Corr_Y @ tr(dx) + epsilon) / (np.conj(dx) @ Corr_Y @ tr(dx) + epsilon)

    # Info: Converting filter from GEFT to STFT
    match freq_mode:
        case 'stft':
            # Info: F_lk_star.shape = [n_bins_star, n_win_F, n_sensors]
            F_lk_star = F_lk
        case 'ssbt':
            F_lk_star = np.empty((n_bins_star, n_win_F, n_sensors), dtype=complex)
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

            Sf_lk_star[k_idx, l_idx] = (he(F) @ S).item()
            If_lk_star[k_idx, l_idx] = (he(F) @ I).item()
            Rf_lk_star[k_idx, l_idx] = (he(F) @ R).item()
            Wf_lk_star[k_idx, l_idx] = (he(F) @ W).item()
            Yf_lk_star[k_idx, l_idx] = (he(F) @ Y).item()

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

    iSINR_lk = np.empty((n_bins_star, n_win_F), dtype=float)
    oSINR_lk = np.empty((n_bins_star, n_win_F), dtype=float)
    gSINR_lk = np.empty((n_bins_star, n_win_F), dtype=float)
    iSINR_k = np.empty((n_bins_star,), dtype=float)
    oSINR_k = np.empty((n_bins_star,), dtype=float)
    gSINR_k = np.empty((n_bins_star,), dtype=float)

    for k_idx in range(n_bins_star):
        for l_idx in range(n_win_F):
            idx_stt = max(0, (l_idx + 1) * dist_fil - n_win_Y)
            idx_end = min((l_idx + 1) * dist_fil, n_win_Y)
            S = S_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)
            W = W_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)

            Sf = Sf_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)
            Wf = Wf_lk_star[k_idx, idx_stt:idx_end].reshape(-1, 1)

            var_S = (he(S) @ S).item()
            var_W = (he(W) @ W).item()

            var_Sf = (he(Sf) @ Sf).item()
            var_Wf = (he(Wf) @ Wf).item()

            iSINR = (var_S + epsilon) / (var_W + epsilon)
            oSINR = (var_Sf + epsilon) / (var_Wf + epsilon)
            iSINR_lk[k_idx, l_idx] = np.real(iSINR)
            oSINR_lk[k_idx, l_idx] = np.real(oSINR)
            gSINR_lk[k_idx, l_idx] = np.real((oSINR + epsilon) / (iSINR + epsilon))

    iSINR_k = np.mean(iSINR_lk, 1)
    oSINR_k = np.mean(oSINR_lk, 1)
    gSINR_k = np.mean(gSINR_lk, 1)

    iSINR_lk = dB(iSINR_lk)
    oSINR_lk = dB(oSINR_lk)
    gSINR_lk = dB(gSINR_lk)
    iSINR_k = dB(iSINR_k)
    oSINR_k = dB(oSINR_k)
    gSINR_k = dB(gSINR_k)

    """
        -------------------
        - Export measures -
        -------------------
    """

    exp_iSINR_lk = ['freq, win, val']
    exp_oSINR_lk =['freq, win, val']
    exp_gSINR_lk =['freq, win, val']
    exp_iSINR_k =['freq, val']
    exp_oSINR_k =['freq, val']
    exp_gSINR_k =['freq, val']

    sym_freqs = sym_freqs/1000
    for k_idx in range(n_bins_star):
        exp_iSINR_k.append(','.join([str(sym_freqs[k_idx]), str(iSINR_k[k_idx])]))
        exp_oSINR_k.append(','.join([str(sym_freqs[k_idx]), str(oSINR_k[k_idx])]))
        exp_gSINR_k.append(','.join([str(sym_freqs[k_idx]), str(gSINR_k[k_idx])]))

        for l_idx in range(n_win_F):
            t = l_idx * dist_fil * n_bins_star / fs
            exp_iSINR_lk.append(','.join([str(sym_freqs[k_idx]), str(t), str(iSINR_lk[k_idx, l_idx])]))
            exp_oSINR_lk.append(','.join([str(sym_freqs[k_idx]), str(t), str(oSINR_lk[k_idx, l_idx])]))
            exp_gSINR_lk.append(','.join([str(sym_freqs[k_idx]), str(t), str(gSINR_lk[k_idx, l_idx])]))

    exp_iSINR_k = '\n'.join(exp_iSINR_k)
    exp_oSINR_k = '\n'.join(exp_oSINR_k)
    exp_gSINR_k = '\n'.join(exp_gSINR_k)
    exp_iSINR_lk = '\n'.join(exp_iSINR_lk)
    exp_oSINR_lk = '\n'.join(exp_oSINR_lk)
    exp_gSINR_lk = '\n'.join(exp_gSINR_lk)

    freq_mode = freq_mode.upper()
    filename = '_' + freq_mode
    folder = 'io_output/' + freq_mode + '/'
    if not os.path.isdir('io_output/'):
        os.mkdir('io_output/')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(folder + 'gain_SINR_k' + filename + '.csv', 'w') as f:
        f.write(exp_gSINR_k)
        f.close()
    with open(folder + 'gain_SINR_lk' + filename + '.csv', 'w') as f:
        f.write(exp_gSINR_lk)
        f.close()

    t_min = 0
    t_max = n_win_Y * n_bins_star / fs

    mesh_cols = r'\def\meshcols{{{}}}'.format(n_win_F)
    mesh_rows = r'\def\meshrows{{{}}}'.format(n_bins_star)
    t_min = r'\def\tmin{{{}}}'.format(t_min)
    t_max = r'\def\tmax{{{}}}'.format(t_max)
    data = [mesh_cols, mesh_rows, t_min, t_max]
    colors = gp.gen_palette(80, 60, ['stft', 'ssbt'], 345)

    data_defs = data + colors
    data_defs = '\n'.join(data_defs)

    with open('io_output/' + 'aux_data.tex', 'w') as f:
        f.write(data_defs)
        f.close()

    plt.imshow(gSINR_lk, interpolation='bicubic')
    plt.show()


def main():
    freq_modes = ['stft',
                  'ssbt']

    for freq_mode in freq_modes:
        simulation(freq_mode=freq_mode,
                   sig_mode='load')


if __name__ == '__main__':
    main()