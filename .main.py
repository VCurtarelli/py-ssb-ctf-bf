import numpy as np
from scipy.signal import stft, istft
from scipy.fft import fft, ifft
from f_ssbt import ssbt, issbt, rft, irft
import matplotlib.pyplot as plt


def main(sim_to_test=1, show=True):

    ## PARAMETERS
    fs = 1000
    nsamples = 2**14
    n_xt = np.arange(0, nsamples)
    t_xt = n_xt / fs

    ## INPUT DATA GENERATION
    data = np.genfromtxt('input_signal_xt.csv')
    xt_1 = data

    ## SIMULATIONS
    match sim_to_test:
        ## TEST 1
        case 1:
            freqs_stft, _, Xlk_stft = stft(xt_1, fs)
            t_xt_stft, xt_stft = istft(Xlk_stft, fs)

            freqs_ssbt, _, Xlk_ssbt = ssbt(xt_1, fs=fs)
            t_xt_ssbt, xt_ssbt = issbt(Xlk_ssbt, fs=fs)

            plt.plot(t_xt, xt_1, linewidth=5, linestyle='-', label='Original')
            plt.plot(t_xt_stft, xt_stft, linewidth=4, linestyle='--', label='STFT')
            plt.plot(t_xt_ssbt, xt_ssbt, linewidth=3, linestyle=':', label='SSBT')
            plt.legend()

            if show is True:
                plt.show()
            else:
                plt.savefig('results_test_1.png')

        ## TEST 2
        case 2:
            freqs_stft, _, Xlk_stft = stft(xt_1, fs)
            freqs_ssbt, _, Xlk_ssbt = ssbt(xt_1, fs=fs)

            # t_ht = t_xt[:256]
            # ht = np.exp(-3*t_ht) + 2*np.random.randn(t_ht.size)
            ht = np.genfromtxt('impulse_response_ht_short.csv')
            Hk_stft = fft(ht)[:129]
            Hk_ssbt = -np.real(fft(ht) * np.exp(-1j*3*np.pi*0.25)) * np.sqrt(2)

            Ylk_stft = 0*Xlk_stft
            Ylk_ssbt = 0*Xlk_ssbt

            for l in range(Xlk_stft.shape[1]):
                Ylk_stft[:, l] = Hk_stft * Xlk_stft[:, l]
                Ylk_ssbt[:, l] = Hk_ssbt * Xlk_ssbt[:, l]

            t_yt_stft, yt_stft = istft(Ylk_stft, fs)
            t_yt_ssbt, yt_ssbt = issbt(Ylk_ssbt, fs=fs)

            plt.plot(t_yt_stft, yt_stft, linewidth=4, linestyle='-', label='STFT')
            plt.plot(t_yt_ssbt, yt_ssbt, linewidth=3, linestyle='-.', label='SSBT')
            plt.plot(t_yt_ssbt, yt_ssbt-yt_stft, linewidth=2, linestyle=':', label='Difference')
            plt.legend()

            if show is True:
                plt.show()
            else:
                plt.savefig('results_test_2.png')

        ## TEST 3
        case 3:
            # ht = np.exp(-3*t_xt) + 2*np.random.randn(t_xt.size)
            # np.savetxt('long_ht.csv', ht, delimiter=',', fmt='%.5f')
            ht = np.genfromtxt('impulse_response_ht_long.csv')

            Xk_fft = fft(xt_1)
            Hk_fft = fft(ht)
            Yk_fft = Xk_fft * Hk_fft
            yt_fft = np.real(ifft(Yk_fft))

            Xk_rft = rft(xt_1)
            Hk_rft = -rft(ht)
            Yk_rft = Xk_rft * Hk_rft
            yt_rft = irft(Yk_rft)

            plt.plot(t_xt, yt_fft, linewidth=4, linestyle='-', label='FT')
            plt.plot(t_xt, yt_rft, linewidth=3, linestyle='-.', label='RFT')
            plt.plot(t_xt, yt_fft-yt_rft, linewidth=2, linestyle=':', label='Diff. RFT-FT')
            plt.legend()

            if show is True:
                plt.show()
            else:
                plt.savefig('results_test_3.png')

        ## TEST 4
        case 4:
            # ht = np.exp(-3*t_xt) + 2*np.random.randn(t_xt.size)
            ht = np.genfromtxt('impulse_response_ht_long.csv')

            Xk_fft = fft(xt_1)
            Hk_fft = fft(ht)
            Yk_fft = Xk_fft * Hk_fft
            yt_fft = np.real(ifft(Yk_fft))

            Xk_rft = rft(xt_1)
            Hk_rft = -rft(ht)
            Yk_rft = Xk_rft * Hk_rft
            yt_rft = irft(Yk_rft)

            Yk_fft_d = np.real(Hk_fft) * Xk_fft + 1j * np.imag(Hk_fft) * np.conj(Xk_fft)
            yt_fft_d = np.real(ifft(Yk_fft_d))

            plt.plot(t_xt, yt_fft, linewidth=4, linestyle='-', label='FT')
            plt.plot(t_xt, yt_rft, linewidth=3, linestyle='-.', label='RFT')
            plt.plot(t_xt, yt_fft_d, linewidth=3, linestyle='--', label='FT dist.')
            plt.plot(t_xt, yt_fft_d-yt_rft, linewidth=2, linestyle='dotted', label='Diff. RFT-FT dist.')
            plt.legend()

            if show is True:
                plt.show()
            else:
                plt.savefig('results_test_4.png')


if __name__ == "__main__":
    """
    Different tests:
    test = 1
        Hypothesis: Check if the SSB and ISSB transforms' implementations work.
        Conclusion: They work.
    
    test = 2
        Hypothesis: Check if the convolution theorem applies to the SSB transform.
        Conclusion: It almost applies, with a small error between the expected result from the STFT transform and the 
        result from SSB transform.
        
    test = 3
        Hypothesis: Check if the convolution theorem applies to the SSB transform when considering the signal as a whole
        (called 'Real Fourier Transform', or RFT).
        Conclusion: It doesn't, now the error between Fourier transform and real Fourier transform are considerable.
    
    test = 4
        Hypothesis: Verify if the result obtained from the mathematical derivations for the SSBT/RFT are correct.
        Conclusion: They are, the results obtained from the RFT and from the mathematical model of it through the
        Fourier transform are identical.    
    """

    test = 4
    main(test)