import os

import numpy as np
import scipy.linalg
from numpy.linalg import inv
from functions import *
import scipy.special as spsp
import scipy as sp
from scipy.signal import stft, istft
from scipy.fft import fft, ifft
from f_ssbt import ssbt, issbt, rft, irft

np.set_printoptions(suppress=True, precision=2)


def simulation(sizes: Params, freq_mode: str = 'stft', tf_mode: str = 'ctf'):
    """
    Parameters
    ----------
    sizes: Params
        Struct of names and sizes of the desired sub-array beamformers.
    name: str
        Name of current simulation.
    freq_mode: str
        Which to use, STFT or SSBT. Defaults to STFT.
    tf_mode: str
        Which to use, CTF or MTF. Defaults to CTF.

    Returns
    -------

    """

    # ---------------
    # Possible values for parameter 'result'

    RESULT_s = {
        'beam': 'Heatmap - Beampattern (abs)',
        'bpt': 'Lineplot - Beampattern (abs and angle_rad)',
        'wng': 'Lineplot - White noise gain (real)',
        'df': 'Lineplot - Directivity factor (real)',
        'snr': 'Lineplot - Signal-to-noise ratio (real)',
        'polar': "Polar plot - Beampatterns for all beamformers",
        'gendata': "Multiplot - Generates 'beam' and 'polar' for the final beamformer"
    }

    # ---------------
    # Sensor array

    Arr_PA = Array(sizes.PA, "PA")

    # ---------------
    # Constants

    dx = 0.018  # x-axis spacing
    c = 343  # wave speed (speed of sound)

    ## ---------------
    ## Signal Info

    # ---------------
    # Desired source
    tx_deg = 0  # desired source direction
    tx_rad = np.deg2rad(tx_deg)
    Var_X = 5  # variance of X

    # ---------------
    # Interfering source
    Ni = 1  # total number of interfering sources
    Var_V = 1
    tv_deg = 60
    tv_rad = np.deg2rad(tv_deg)

    Var_R = 1

    ## ---------------
    ## Simulation Info

    apoints = 21
    nperseg = 32
    fs = 8000
    sym_freqs = np.linspace(0, fs, nperseg + 1)[:(1 + nperseg) // 2 + 1]
    sym_angles = np.linspace(0, 180, apoints)
    sym_angles = np.concatenate([sym_angles, - 180 + sym_angles[1:-1], [-180]])

    sym_angles = list(set(sym_angles))
    sym_angles.sort()

    sym_angles = np.sort(sym_angles)

    params = Params(dx=dx, dy=dx,
                    sym_freqs=sym_freqs, sym_angles=sym_angles,
                    td_rad=tx_rad, c=c, aleph=1)

    ## ---------------
    ## Array matrices
    Arr_PA.calc_vals(params, Arr_PA)  # Calculates position for all sensors of sub-array
    Arr_PA.init_metrics(sym_freqs.size, sym_angles.size, Ni)  # Initializes metrics

    ## ---------------
    ## Impulse Response
    n = np.arange(8000)
    Hn = np.exp(-200 * n / fs)
    n = n[Hn >= 0.01]
    Hn = Hn[Hn >= 0.01]
    f = np.linspace(0, fs, n.size)
    Hlk_ = []
    Glk_ = []
    Tlk_angles = []
    freqs_t = []
    Hlk_true_ = []
    Glk_true_ = []
    for m in range(Arr_PA.M):
        sv_tx_f = calcSV(Arr_PA.r[m], Arr_PA.p[m], tx_rad, f, c)
        freqs_t, Hlk_m = calcRev(sv_tx_f, Hn, fs, freq_mode=freq_mode)
        Hlk_.append(Hlk_m)

        sv_tv_f = calcSV(Arr_PA.r[m], Arr_PA.p[m], tv_rad, f, c)
        _, Glk_m = calcRev(sv_tv_f, Hn, fs, freq_mode=freq_mode)
        Glk_.append(Glk_m)

        freqs_t_true, Hlk_m_true = calcRev(sv_tx_f, Hn, fs, freq_mode='stft')
        _, Glk_m_true = calcRev(sv_tv_f, Hn, fs, freq_mode='stft')
        Hlk_true_.append(Hlk_m_true)
        Glk_true_.append(Glk_m_true)

        Tlk_ = []
        for a_idx, angle in enumerate(sym_angles):
            sv_ta_f = calcSV(Arr_PA.r[m], Arr_PA.p[m], np.deg2rad(angle), f, c)
            _, Tlk_m = calcRev(sv_ta_f, Hn, fs, freq_mode='stft')
            Tlk_.append(Tlk_m)
        Tlk = np.zeros(Tlk_[0].shape + (sym_angles.size,), dtype=complex)
        for a_idx, angle in enumerate(sym_angles):
            Tlk[:, :, a_idx] = Tlk_[a_idx]
        Tlk_angles.append(Tlk)

    Hlk = np.zeros(Hlk_[0].shape + (Arr_PA.M,), dtype=complex)
    Glk = np.zeros(Glk_[0].shape + (Arr_PA.M,), dtype=complex)
    Hlk_true = np.zeros(Hlk_true_[0].shape + (Arr_PA.M,), dtype=complex)
    Glk_true = np.zeros(Glk_true_[0].shape + (Arr_PA.M,), dtype=complex)
    Tlk = np.zeros(Tlk_angles[0].shape + (Arr_PA.M,), dtype=complex)
    for m in range(Arr_PA.M):
        Hlk[:, :, m] = Hlk_[m]
        Glk[:, :, m] = Glk_[m]
        Hlk_true[:, :, m] = Hlk_true_[m]
        Glk_true[:, :, m] = Glk_true_[m]
        Tlk[:, :, :, m] = Tlk_angles[m]

    ## ---------------
    ## Frequency sims
    Zk_ = np.zeros([freqs_t.size, Arr_PA.M], dtype=complex)
    for k_idx, k in enumerate(sym_freqs):
        ## ---------------
        ## Variable reorganization
        Hk = Hlk[k_idx, :, :]
        Gk = Glk[k_idx, :, :]
        dk_x = Hk[0, :]
        Hk_ = Hk[1:, :]

        ## ---------------
        ## Undesired signal correlation matrix
        Corr_w_true = np.zeros([Arr_PA.M, Arr_PA.M], dtype=complex)
        for idx_l in range(Hk_.shape[0]):
            pH_l = Hk_[idx_l, :].reshape(-1, 1)
            Corr_w_true += (pH_l @ he(pH_l)) * Var_X
        for idx_l in range(Gk.shape[0]):
            pG_l = Gk[idx_l, :].reshape(-1, 1)
            Corr_w_true += (pG_l @ he(pG_l)) * Var_V
        Corr_w_true += np.identity(Arr_PA.M) * Var_R
        iCorr_w_true = inv(Corr_w_true)

        Corr_w_true_true = np.zeros([Arr_PA.M, Arr_PA.M], dtype=complex)
        for idx_l in range(1, Hk.shape[0]):
            pH_l = Hlk_true[k_idx, idx_l, :].reshape(-1, 1)
            Corr_w_true_true += (pH_l @ he(pH_l)) * Var_X
        for idx_l in range(Gk.shape[0]):
            pG_l = Glk_true[k_idx, idx_l, :].reshape(-1, 1)
            Corr_w_true_true += (pG_l @ he(pG_l)) * Var_V
        Corr_w_true_true += np.identity(Arr_PA.M) * Var_R

        print(Corr_w_true)
        print(Corr_w_true_true)
        match tf_mode:
            case 'mtf':
                pG_0 = Gk[0, :].reshape(-1, 1)
                Corr_w_tf = (pG_0 @ he(pG_0)) * Var_V + np.identity(Arr_PA.M) * Var_R
                iCorr_w_tf = inv(Corr_w_tf)
            case 'ctf':
                Corr_w_tf = Corr_w_true
                iCorr_w_tf = iCorr_w_true
        ## ---------------
        ## Beamforming
        zk_mvdr = (iCorr_w_tf @ dk_x) / (he(dk_x) @ iCorr_w_tf @ dk_x)
        Zk_[k_idx, :] = zk_mvdr

        Arr_PA.d_x = dk_x
        Arr_PA.h = zk_mvdr

        if freq_mode == 'ssbt':
            ## ---------------
            ## Variable reorganization
            Hk = Hlk[-k_idx, :, :]
            Gk = Glk[-k_idx, :, :]
            dk_x = Hk[0, :]
            Hk_ = Hk[1:, :]

            ## ---------------
            ## Undesired signal correlation matrix
            Corr_w_true = np.zeros([Arr_PA.M, Arr_PA.M], dtype=complex)
            for idx_l in range(Hk_.shape[0]):
                pH_l = Hk_[idx_l, :].reshape(-1, 1)
                Corr_w_true += (pH_l @ he(pH_l)) * Var_X
            for idx_l in range(Gk.shape[0]):
                pG_l = Gk[idx_l, :]
                Corr_w_true += (pG_l @ he(pG_l)) * Var_V
            Corr_w_true += np.identity(Arr_PA.M) * Var_R
            iCorr_w_true = inv(Corr_w_true)

            match tf_mode:
                case 'mtf':
                    pG_0 = Gk[0, :].reshape(-1, 1)
                    Corr_w_tf = (pG_0 @ he(pG_0)) * Var_V + np.identity(Arr_PA.M) * Var_R
                    iCorr_w_tf = inv(Corr_w_tf)
                case 'ctf':
                    Corr_w_tf = Corr_w_true
                    iCorr_w_tf = iCorr_w_true
            ## ---------------
            ## Beamforming
            zk_mvdr_ = (iCorr_w_tf @ dk_x) / (he(dk_x) @ iCorr_w_tf @ dk_x)
            Zk_[k_idx, :] = zk_mvdr

            zk_mvdr = (zk_mvdr + np.conj(zk_mvdr_))/2
            dk_x = Hlk_true[k_idx, 0, :]

        ## ---------------
        ## Metrics
        for a_idx, angle in enumerate(sym_angles):
            d_ta = Tlk[k_idx, 0, a_idx, :]
            bpt = np.abs(calcbeam(Arr_PA.h, d_ta))
            Arr_PA.beam[k_idx, a_idx] = bpt

        Arr_PA.wng[k_idx] = calcwng(zk_mvdr, dk_x)
        Arr_PA.gain[k_idx] = calcgain(zk_mvdr, dk_x, Corr_w_true_true, Corr_w_true_true[0, 0])
        Arr_PA.df[k_idx] = calcdf(Arr_PA, k, c)

        ## ---------------
        ## Result presentation
        freqs = sym_freqs.reshape(-1, ) / 1000
        angles = sym_angles.reshape(-1, )

        beam = dB(vect(Arr_PA.beam).reshape(-1, ))
        wng = dB(Arr_PA.wng.reshape(-1, ))
        gain = dB(Arr_PA.gain.reshape(-1, ))
        df = dB(Arr_PA.df.reshape(-1, ))

    params = [freqs, angles, wng, df, beam, gain]
    params = fixDec(params)
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
    filename = 'res' + '_' + tf_mode + '_' + freq_mode + '_'
    folder = 'results/' + tf_mode + '_' + freq_mode + '/'
    if not os.path.isdir('results/'):
        os.mkdir('results/')
    if not os.path.isdir(folder):
        os.mkdir(folder)
    with open(folder + filename + 'wng.csv', 'w') as f:
        f.write(wng_)
        f.close()
    with open(folder + filename + 'df.csv', 'w') as f:
        f.write(df_)
        f.close()
    with open(folder + filename + 'gain.csv', 'w') as f:
        f.write(gain_)
        f.close()
    with open(folder + filename + 'beam.csv', 'w') as f:
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

    with open(folder + 'data_defs.tex', 'w') as f:
        f.write(data_defs)
        f.close()


def main():
    sizes = {
        'A': Params(PA=(1, 5, 0)),
    }

    for name in sizes.keys():
        size = sizes[name]
        simulation(size,
                   freq_mode='stft',
                   tf_mode='ctf')


if __name__ == '__main__':
    main()
