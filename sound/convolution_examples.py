import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

import librosa
import librosa.display
import soundfile as sf

def plot_convolution_as_filtering(sr=44100, cutoff=1000.0, num_taps=101):
    """
    Demonstrates that filtering = convolution in time domain
                                 = multiplication in frequency domain.
    This is the convolution theorem — a core DSP concept.
    """
    # Design a simple low-pass FIR filter
    nyquist      = sr / 2
    h            = firwin(num_taps, cutoff / nyquist)   # filter kernel
    w, H         = freqz(h, worN=8000, fs=sr)

    # Synthetic signal: two tones, one below and one above cutoff
    duration     = 1.0
    t            = np.linspace(0, duration, sr)
    freq_low     = 300.0    # will pass through
    freq_high    = 3000.0   # will be removed
    y            = (np.sin(2 * np.pi * freq_low  * t) +
                    np.sin(2 * np.pi * freq_high * t))
    y_filtered   = np.convolve(y, h, mode='same')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Convolution Theorem: Filtering in Time = Multiplication in Frequency',
                 fontsize=13, fontweight='bold')

    # Filter kernel (time domain)
    axes[0, 0].plot(h, color='#9b59b6')
    axes[0, 0].set_title('① Filter Kernel h[n] (time domain)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.4)

    # Filter frequency response
    axes[0, 1].plot(w, 20 * np.log10(np.abs(H) + 1e-10), color='#9b59b6')
    axes[0, 1].axvline(x=cutoff, color='red', linestyle='--',
                       label=f'Cutoff ({cutoff} Hz)')
    axes[0, 1].set_title('② Filter Frequency Response H(f)')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_xlim(0, 5000)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.4)

    # Original vs filtered waveform
    axes[1, 0].plot(t[:1000], y[:1000],          color='#4a90d9', alpha=0.7, label='Original')
    axes[1, 0].plot(t[:1000], y_filtered[:1000], color='#e74c3c', label='Filtered')
    axes[1, 0].set_title(f'③ Waveform Before & After Convolution\n'
                         f'({freq_low} Hz tone survives, {freq_high} Hz tone removed)')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.4)

    # FFT before and after
    fft_orig     = np.abs(np.fft.rfft(y))
    fft_filtered = np.abs(np.fft.rfft(y_filtered))
    freqs        = np.fft.rfftfreq(len(y), 1 / sr)
    axes[1, 1].plot(freqs, fft_orig,     color='#4a90d9', alpha=0.7, label='Original')
    axes[1, 1].plot(freqs, fft_filtered, color='#e74c3c', label='Filtered')
    axes[1, 1].axvline(x=cutoff, color='black', linestyle='--',
                       label=f'Cutoff ({cutoff} Hz)')
    axes[1, 1].set_title('④ Frequency Spectrum Before & After')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_xlim(0, 5000)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_convolution_as_filtering()