import os

import numpy as np
import scipy.linalg
from numpy.linalg import inv, det
from functions import *
import scipy.special as spsp
import scipy as sp
from scipy.signal import stft, istft
from scipy.fft import fft, ifft
from f_ssbt import ssbt, issbt, rft, irft

np.set_printoptions(suppress=True, precision=4)


def simulation(sizes: Params, freq_mode: str = 'stft', tf_mode: str = 'ctf', res_mode='save'):
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
    res_mode: str
        Which results to make, plot:metric or save-to-file. Defaults to save-to-file ('save').
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

    dx = 0.015  # x-axis spacing
    c = 343  # wave speed (speed of sound)

    # # ---------------
    # # Signal Info

    # ---------------
    # Desired source
    tx_deg = 0  # desired source direction
    tx_rad = np.deg2rad(tx_deg)
    Var_X = 1  # variance of X

    # ---------------
    # Interfering source
    Ni = 1  # total number of interfering sources
    Var_V = 1
    tv_deg = 60
    tv_rad = np.deg2rad(tv_deg)

    Var_R = 1e-4

    # # ---------------
    # # Simulation Info

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

    # # ---------------
    # # Array matrices
    Arr_PA.calc_vals(params, Arr_PA)  # Calculates position for all sensors of sub-array
    Arr_PA.init_metrics(sym_freqs.size, sym_angles.size, Ni)  # Initializes metrics

    # # ---------------
    # # Impulse Response
    n = np.arange(8000)
    Hn = np.exp(-200 * n / fs)
    n = n[Hn >= 0.01]
    Hn = Hn[Hn >= 0.01]
    Hn[0] += 1

    #    # # ---------------
    #    # # TESTING - DELETE LATER
    #    # # ---------------
    #    Hn[0] = 1
    #    Hn[1:] = 0
    #    # # ---------------
    #    # # TESTING - DELETE LATER
    #    # # ---------------

    f = np.linspace(0, fs, n.size)
    Hlk_ = []
    Glk_ = []
    Tlk_angles = []
    freqs_t = []

    sv_tx_f = calcSV(Arr_PA.r[0], Arr_PA.p[0], tx_rad, f, c)

    freqs_t, Hlk_m = calcRev(sv_tx_f, Hn, fs, freq_mode)
    Hlk = np.zeros(Hlk_m.shape + (Arr_PA.M,), dtype=complex)
    Glk = np.zeros(Hlk_m.shape + (Arr_PA.M,), dtype=complex)

    freqs_t_true, Hlk_m_true = calcRev(sv_tx_f, Hn, fs, 'stft')
    Hlk_true = np.zeros(Hlk_m_true.shape + (Arr_PA.M,), dtype=complex)
    Glk_true = np.zeros(Hlk_m_true.shape + (Arr_PA.M,), dtype=complex)
    Tlk_true = np.zeros(Hlk_m_true.shape + (sym_angles.size, Arr_PA.M,), dtype=complex)

    for m in range(Arr_PA.M):
        sv_tx_f = calcSV(Arr_PA.r[m], Arr_PA.p[m], tx_rad, f, c)
        freqs_t, Hlk_m = calcRev(sv_tx_f, Hn, fs, freq_mode)

        sv_tv_f = calcSV(Arr_PA.r[m], Arr_PA.p[m], tv_rad, f, c)
        freqs_t, Glk_m = calcRev(sv_tv_f, Hn, fs, freq_mode)

        freqs_t_true, Hlk_m_true = calcRev(sv_tx_f, Hn, fs, 'stft')
        freqs_t_true, Glk_m_true = calcRev(sv_tv_f, Hn, fs, 'stft')

        Hlk[:, :, m] = Hlk_m
        Glk[:, :, m] = Glk_m
        Hlk_true[:, :, m] = Hlk_m_true
        Glk_true[:, :, m] = Glk_m_true
        for a_idx, angle in enumerate(sym_angles):
            sv_ta_f = calcSV(Arr_PA.r[m], Arr_PA.p[m], np.deg2rad(angle), f, c)
            _, Tlk_m_true = calcRev(sv_ta_f, Hn, fs, 'stft')
            Tlk_true[:, :, a_idx, m] = Tlk_m_true

    # print(Hlk_m)
    # print()
    # print(Glk_m)
    # print()
    # print(Hlk_m - Glk_m)
    # print()
    # print(sv_tx_f - sv_tv_f)
    # input()

    # print(sym_angles[20])
    # print(Hlk[0,0,0])
    # print(Glk[0,0,0])
    # print(Hlk_true[0,0,0])
    # print(Glk_true[0,0,0])
    # print(Tlk_true[0,0,20,0])
    # input()
    # Hlk = Hlk / Hlk[0, 0, 0]
    # Glk = Glk / Hlk[0, 0, 0]
    # Hlk_true = Hlk_true / Hlk_true[0, 0, 0]
    # Glk_true = Glk_true / Hlk_true[0, 0, 0]
    # Tlk_true = Tlk_true / Hlk_true[0, 0, 0]
    # # ---------------
    # # Frequency sims
    Zk = np.zeros([freqs_t.size, Arr_PA.M], dtype=complex)
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
            Corr_w += (pH_l @ he(pH_l)) * Var_X
        for idx_l in range(Gk.shape[0]):
            pG_l = Gk[idx_l, :]
            Corr_w += (pG_l @ he(pG_l)) * Var_V
        Corr_w += np.identity(Arr_PA.M) * Var_R
        #        print(Corr_w)
        #        print(det(Corr_w))
        #        input()
        iCorr_w = inv(Corr_w)

        match tf_mode:
            case 'mtf':
                pG_0 = Gk[0, :].reshape(-1, 1)
                Corr_w_tf = (pG_0 @ he(pG_0)) * Var_V + np.identity(Arr_PA.M) * Var_R
                iCorr_w_tf = inv(Corr_w_tf)
            case 'ctf':
                Corr_w_tf = Corr_w
                iCorr_w_tf = iCorr_w
        # # ---------------
        # # Beamforming
        zk_mvdr = (iCorr_w_tf @ dk_x) / (he(dk_x) @ iCorr_w_tf @ dk_x)
        Zk[k_idx, :] = zk_mvdr

    for m in range(Arr_PA.M):
        # # ---------------
        # # Frequency-transform correction
        zk_m = Zk[:, m]
        if freq_mode == 'stft':
            zk_m_true = zk_m
        else:
            zn_m = irft(zk_m)
            zk_m_true = fft(zn_m)[:freqs_t_true.size]
        Zk_true[:, m] = zk_m_true
    Hlk = Hlk_true
    Glk = Glk_true
    Zk = Zk_true
    freqs_t = freqs_t_true

    for k_idx, k in enumerate(freqs_t):

        # # ---------------
        # # Variable reorganization
        Hk = Hlk[k_idx, :, :]
        Gk = Glk[k_idx, :, :]
        dk_x = Hk[0, :]
        Hk_ = Hk[1:, :]
        zk_mvdr = Zk[k_idx, :]

        # # ---------------
        # # Undesired signal correlation matrix
        Corr_w = np.zeros([Arr_PA.M, Arr_PA.M], dtype=complex)
        for idx_l in range(Hk_.shape[0]):
            pH_l = Hk_[idx_l, :].reshape(-1, 1)
            Corr_w += (pH_l @ he(pH_l)) * Var_X
        for idx_l in range(Gk.shape[0]):
            pG_l = Gk[idx_l, :]
            Corr_w += (pG_l @ he(pG_l)) * Var_V
        Corr_w += np.identity(Arr_PA.M) * Var_R
        iCorr_w = inv(Corr_w)

        # # ---------------
        # # Metrics
        for a_idx, angle in enumerate(sym_angles):
            d_ta = Tlk_true[k_idx, 0, a_idx, :]
            bpt = np.abs(calcbeam(zk_mvdr, d_ta))
            Arr_PA.beam[k_idx, a_idx] = bpt

        Arr_PA.wng[k_idx] = calcwng(zk_mvdr, dk_x)
        Arr_PA.gain[k_idx] = calcgain(zk_mvdr, dk_x, Corr_w, Corr_w[0, 0])
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
            filename = '_' + tf_mode + '_' + freq_mode
            folder = 'results/' + tf_mode + '_' + freq_mode + '/'
            if not os.path.isdir('results/'):
                os.mkdir('results/')
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

            with open('results/' + 'data_defs.tex', 'w') as f:
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
        simulation(Params(PA=(1, 10, 0)),
                   freq_mode=modes.freq_mode,
                   tf_mode=modes.tf_mode,
                   res_mode='save')


if __name__ == '__main__':
    main()
