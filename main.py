import os

import numpy as np
import scipy.linalg
from numpy.linalg import inv, det
from functions import *
import scipy.special as spsp
import scipy as sp
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.fft import fft, ifft
from f_ssbt import ssbt, issbt, rft, irft
import gen_palette as gp
import soundfile as sf
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool


np.set_printoptions(suppress=True, precision=6, threshold=sys.maxsize)


def sim_parser(comb):
    simulation(freq_mode=comb[0], sig_mode='load', n_per_seg=comb[1])


def simulation(freq_mode: str = 'stft', sig_mode='random', n_per_seg=32):
    """
    Parameters
    ----------
    freq_mode: str
        Which to use, STFT or SSBT. Defaults to STFT.
    sig_mode: str
        Which signals to use, 'random' or 'load'. Defaults to 'random'.
    n_per_seg: int
        Number of samples per window in transforms. Defaults to 32.
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

    global n_bins, F_lk_star, geft, x_n, v_n, r_n
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
    win_p_fil = 25  # INFO: If "win_p_fil = -1", uses whole signal
    m_ref = 0
    array = Array((1, n_sensors), 'Array')
    array.init_metrics(sym_freqs.size)  # Initializes metrics

    epsilon = 1e-15
    iSIR = 5
    iSNR = 90

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
            x_n, samplerate = sf.read('io_input/audio_speech_male_r.flac')
            x_n = resample(x_n, samplerate, fs)
            babble = scipy.io.loadmat('io_input/babble.mat')
            v_n, samplerate = babble['babble'].reshape(-1), babble['fs']
            v_n = resample(v_n, samplerate, fs)
            r_n, samplerate = sf.read('io_input/audio_noise_wgn.flac')
            r_n = resample(r_n, samplerate, fs).reshape(-1, 1)
            nr_n = []
            for m in range(n_sensors):
                nr_n.append(np.roll(r_n, int(m/n_sensors * r_n.size)))
            r_n = np.hstack(nr_n)

        case 'random', _:
            len_x = 100000
            x_n = np.random.rand(len_x)
            v_n = np.random.rand(2*len_x)
            r_n = np.random.rand(2*len_x, n_sensors)

    """
        ------------------------------
        - Relative-impulse responses -
        ------------------------------
    """

    H_f = np.empty_like(h_n, dtype=complex)
    G_f = np.empty_like(g_n, dtype=complex)
    for m in range(n_sensors):
        H_f[m, :] = fft(h_n[m, :])
        G_f[m, :] = fft(g_n[m, :])

    B_f = H_f / H_f[m_ref, :]
    C_f = G_f / G_f[m_ref, :]

    b_n = np.empty_like(B_f, dtype=float)
    c_n = np.empty_like(C_f, dtype=float)
    for m in range(n_sensors):
        b_n[m, :] = np.real(ifft(B_f[m, :]))
        c_n[m, :] = np.real(ifft(C_f[m, :]))

    x1_n = np.convolve(h_n[m_ref, :], x_n, mode='full')
    v1_n = np.convolve(g_n[m_ref, :], v_n, mode='full')

    # INFO: Array-fixing, so that the desired signal RIR's max. value is at the start of a FT window

    idx_max_b_n = np.where(np.abs(b_n[m_ref, :]) == np.abs(np.amax(b_n[m_ref, :])))[0][0]
    new_idx_max_b_n = int(np.ceil(idx_max_b_n / (n_per_seg//2)) * (n_per_seg // 2))
    b_n = np.hstack([np.zeros([n_sensors, new_idx_max_b_n - idx_max_b_n]), b_n])
    c_n = np.hstack([np.zeros([n_sensors, new_idx_max_b_n - idx_max_b_n]), c_n])

    l_des_window = new_idx_max_b_n // (n_per_seg // 2)

    len_rir = b_n.shape[1]
    n_win_rir = int(np.ceil(2*len_rir/n_per_seg + 1))
    n_win_R = int(np.ceil(2*r_n.shape[0]/n_per_seg + 1))

    """
        -------------------------------
        - Time-Freq. domain variables -
        -------------------------------
    """

    # Vars: Sources and signals (freq.)
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

    B_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    C_lk = np.empty((n_bins, n_win_rir, n_sensors), dtype=complex)
    N_lk = np.empty((n_bins, n_win_R, n_sensors), dtype=complex)
    for idx in range(n_sensors):
        _, _, Bm_lk = geft(b_n[idx, :], fs, nperseg=n_per_seg)
        B_lk[:, :, idx] = Bm_lk

        _, _, Cm_lk = geft(c_n[idx, :], fs, nperseg=n_per_seg)
        C_lk[:, :, idx] = Cm_lk

        _, _, Rm_lk = geft(r_n[:, idx], fs, nperseg=n_per_seg)
        N_lk[:, :, idx] = Rm_lk

    P_lk = np.copy(B_lk)
    P_lk[:, l_des_window, :] = 0
    dx_lk = B_lk - P_lk
    dx_k = B_lk[:, l_des_window, :]

    _, _, X1_lk = geft(x1_n, fs, nperseg=n_per_seg)
    _, _, V1_lk = geft(v1_n, fs, nperseg=n_per_seg)

    n_win_X1 = X1_lk.shape[1]
    n_win_V1 = V1_lk.shape[1]
    n_win_R = N_lk.shape[1]
    n_win_Y = n_win_rir+n_win_X1-1

    if max(n_win_V1, n_win_R) < n_win_X1 + n_win_rir-1:
        raise AssertionError('Noise signals too short.')

    S_lk = np.empty((n_bins, n_win_rir+n_win_X1-1, n_sensors), dtype=complex)
    W_lk = np.empty((n_bins, n_win_rir+n_win_X1-1, n_sensors), dtype=complex)
    Y_lk = np.empty((n_bins, n_win_rir+n_win_X1-1, n_sensors), dtype=complex)
    for m in range(n_sensors):
        for k_idx in range(n_bins):
            # INFO: CTF convolutions and signal-length correction
            S = np.convolve(dx_lk[k_idx, :, m], X1_lk[k_idx, :], mode='full')[:n_win_Y]
            U = np.convolve(P_lk[k_idx, :, m], X1_lk[k_idx, :], mode='full')[:n_win_Y]
            I = np.convolve(C_lk[k_idx, :, m], V1_lk[k_idx, :], mode='full')[:n_win_Y]
            R = (N_lk[k_idx, :, m])[:n_win_Y]

            # INFO: Variance and SIR/SNR calculations
            var_S = np.var(S)
            U = U / np.sqrt(var_S + epsilon)
            S = S / np.sqrt(var_S + epsilon)
            I = I / np.sqrt(np.var(I) + epsilon) / np.sqrt(10 ** (iSIR / 10))
            R = R / np.sqrt(np.var(R) + epsilon) / np.sqrt(10 ** (iSNR / 10))
            W = U + I + R

            # INFO: Calculating desired, undesired, and observed signals
            S_lk[k_idx, :, m] = S
            W_lk[k_idx, :, m] = W
            Y_lk[k_idx, :, m] = S + W

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

    B_lk_star = np.empty((n_bins_star, n_win_rir, n_sensors), dtype=complex)
    C_lk_star = np.empty((n_bins_star, n_win_rir, n_sensors), dtype=complex)
    N_lk_star = np.empty((n_bins_star, n_win_R, n_sensors), dtype=complex)
    for idx in range(n_sensors):
        _, _, Bm_lk_star = stft(b_n[idx, :], fs, nperseg=n_per_seg)
        B_lk_star[:, :, idx] = Bm_lk_star

        _, _, Cm_lk_star = stft(c_n[idx, :], fs, nperseg=n_per_seg)
        C_lk_star[:, :, idx] = Cm_lk_star

        _, _, Rm_lk_star = stft(r_n[:, idx], fs, nperseg=n_per_seg)
        N_lk_star[:, :, idx] = Rm_lk_star

    P_lk_star = np.copy(B_lk_star)
    P_lk_star[:, l_des_window, :] = 0
    dx_lk_star = B_lk_star - P_lk_star
    dx_k_star = B_lk_star[:, l_des_window, :]

    _, _, X1_lk_star = stft(x1_n, fs, nperseg=n_per_seg)
    _, _, V1_lk_star = stft(v1_n, fs, nperseg=n_per_seg)

    S_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X1-1, n_sensors), dtype=complex)
    W_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X1-1, n_sensors), dtype=complex)
    Y_lk_star = np.empty((n_bins_star, n_win_rir+n_win_X1-1, n_sensors), dtype=complex)
    for m in range(n_sensors):
        for k_idx in range(n_bins_star):
            # INFO: CTF convolutions and signal-length correction
            S = np.convolve(dx_lk_star[k_idx, :, m], X1_lk_star[k_idx, :], mode='full')[:n_win_Y]
            U = np.convolve(P_lk_star[k_idx, :, m], X1_lk_star[k_idx, :], mode='full')[:n_win_Y]
            I = np.convolve(C_lk_star[k_idx, :, m], V1_lk_star[k_idx, :], mode='full')[:n_win_Y]
            R = (N_lk_star[k_idx, :, m])[:n_win_Y]

            # INFO: Variance and SIR/SNR calculations
            var_S = np.var(S)
            U = U / np.sqrt(var_S + epsilon)
            S = S / np.sqrt(var_S + epsilon)
            I = I / np.sqrt(np.var(I) + epsilon) / np.sqrt(10 ** (iSIR / 10))
            R = R / np.sqrt(np.var(R) + epsilon) / np.sqrt(10 ** (iSNR / 10))
            W = U + I + R

            # INFO: Calculating desired, undesired, and observed signals
            S_lk_star[k_idx, :, m] = S
            W_lk_star[k_idx, :, m] = W
            Y_lk_star[k_idx, :, m] = S + W

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
        D = dx_k[k_idx, :]
        D = D.reshape(-1, 1)
        for l_idx in range(n_win_F):
            # INFO: Separating necessary windows of Y_lk, and calculating coherence matrix
            idx_stt = max(0, (l_idx+1)*dist_fil-win_p_fil)
            idx_end = min((l_idx+1)*dist_fil, n_win_Y)
            Y = W_lk[k_idx, idx_stt:idx_end, :]
            Corr_Y = np.empty([n_sensors, n_sensors], dtype=complex)
            for idx_i in range(n_sensors):
                for idx_j in range(idx_i, n_sensors):
                    Yi = Y[:, idx_i].reshape(-1, 1)
                    Yj = Y[:, idx_j].reshape(-1, 1)
                    Corr_Y[idx_i, idx_j] = (he(Yi) @ Yj).item()
                    Corr_Y[idx_j, idx_i] = np.conj(Corr_Y[idx_i, idx_j])
            try:
                iCorr_Y = inv(Corr_Y)
                F_lk[k_idx, l_idx, :] = ((iCorr_Y @ D) / (he(D) @ iCorr_Y @ D + epsilon)).reshape(n_sensors)
            except np.linalg.LinAlgError:
                F_lk[k_idx, l_idx, :] = 0

    # Info: Assuring filter is in STFT
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
    Wf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    Yf_lk_star = np.empty((n_bins_star, n_win_Y), dtype=complex)
    
    for k_idx in range(n_bins_star):
        for l_idx in range(n_win_Y):
            F = F_lk_star[k_idx, l_idx//dist_fil, :].reshape(-1, 1)
            S = S_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            W = W_lk_star[k_idx, l_idx, :].reshape(-1, 1)
            Y = Y_lk_star[k_idx, l_idx, :].reshape(-1, 1)

            Sf_lk_star[k_idx, l_idx] = (he(F) @ S).item()
            Wf_lk_star[k_idx, l_idx] = (he(F) @ W).item()
            Yf_lk_star[k_idx, l_idx] = (he(F) @ Y).item()

    """
        -----------
        - Metrics -
        -----------
    """

    # Vars: Sources and signals (freq.)
    #       gSINR_lk    : Narrowband gain in SNR per-window [scalar, dB]
    #       gSINR_k     : Narrowband window-average gain in SNR [scalar, dB]
    #       dsdi_lk     : Narrowband desired-signal distortion index [scalar]

    gSINR_lk = np.empty((n_bins_star, n_win_F), dtype=float)
    DSDI_lk = np.empty((n_bins_star, n_win_F), dtype=float)

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
            gSINR_lk[k_idx, l_idx] = np.real((oSINR + epsilon) / (iSINR + epsilon))

            F = F_lk_star[k_idx, l_idx // dist_fil, :].reshape(-1, 1)
            D = dx_k_star[k_idx, :].reshape(-1, 1)
            DSDI_lk[k_idx, l_idx] = (np.abs(he(F)@D - 1)**2).item()

    gSINR_k = np.mean(gSINR_lk, 1)

    gSINR_lk = dB(gSINR_lk)
    gSINR_k = dB(gSINR_k)

    """
        ---------------
        - Export data -
        ---------------
    """

    exp_gSINR_lk =['freq, win, val']
    exp_gSINR_k =['freq, val']
    exp_DSDI_lk = ['freq, win, val']

    sym_freqs = sym_freqs/1000
    for k_idx in range(n_bins_star):
        exp_gSINR_k.append(','.join([str(sym_freqs[k_idx]), str(gSINR_k[k_idx])]))

        for l_idx in range(n_win_F):
            t = l_idx * dist_fil * n_bins_star / fs
            exp_gSINR_lk.append(','.join([str(sym_freqs[k_idx]), str(t), str(gSINR_lk[k_idx, l_idx])]))
            exp_DSDI_lk.append(','.join([str(sym_freqs[k_idx]), str(t), str(DSDI_lk[k_idx, l_idx])]))

    exp_gSINR_k = '\n'.join(exp_gSINR_k)
    exp_gSINR_lk = '\n'.join(exp_gSINR_lk)
    exp_DSDI_lk = '\n'.join(exp_DSDI_lk)

    freq_mode = freq_mode.upper()
    filename = '_' + freq_mode + '_' + str(n_per_seg)
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
    with open(folder + 'DSDI_lk' + filename + '.csv', 'w') as f:
        f.write(exp_DSDI_lk)
        f.close()

    _, yf_n = istft(Yf_lk_star, fs)
    yf_n = 0.9 * yf_n/np.amax(yf_n)
    wavfile.write('io_output/audios/yf_' + freq_mode + '_' + str(n_per_seg) + '.wav', fs, yf_n)

    _, y_n = istft(Y_lk_star[:, :, 0], fs)
    y_n = 0.9 * y_n/np.amax(y_n)
    wavfile.write('io_output/audios/y1_unfiltered.wav', fs, y_n)

    _, x1_n = istft(X1_lk_star, fs)
    x1_n = 0.9 * x1_n/np.amax(x1_n)
    wavfile.write('io_output/audios/x1_unfiltered.wav', fs, x1_n)
    """
        --------------------
        - Export aux files -
        --------------------
    """

    t_min = 0
    t_max = n_win_Y * n_bins_star / fs

    mesh_cols = r'\def\meshcols{{{}}}'.format(n_win_F)
    mesh_rows = r'\def\meshrows{{{}}}'.format(n_bins_star)
    t_min = r'\def\tmin{{{}}}'.format(t_min)
    t_max = r'\def\tmax{{{}}}'.format(t_max)
    data = [mesh_cols, mesh_rows, t_min, t_max]
    data_defs = '\n'.join(data)

    with open('io_output/' + 'aux_data_' + str(n_per_seg) + '.tex', 'w') as f:
        f.write(data_defs)
        f.close()

    colors = gp.gen_palette(80, 60, ['A', 'B', 'C', 'D', 'E', 'F'], 345)
    color_defs = '\n'.join(colors)

    with open('io_output/' + 'colors_' + str(len(colors)) + '.tex', 'w') as f:
        f.write(color_defs)
        f.close()

    print(freq_mode, n_per_seg)
    return None


def main():
    freqmodes = ['stft',
                 'ssbt']

    npersegs = [32, 64, 128]

    combs = [(freqmode, nperseg) for freqmode in freqmodes for nperseg in npersegs]
    ncombs = len(combs)
    # idx = 0
    with Pool(ncombs) as p:
        p.map(sim_parser, combs)
    # for freq_mode in freqmodes:
    #     for nperseg in npersegs:
    #         simulation(freq_mode=freq_mode,
    #                    sig_mode='load',
    #                    n_per_seg=nperseg)
    #         idx += 1
    #         print(str(idx) + '/' + str(ncombs))


if __name__ == '__main__':
    main()