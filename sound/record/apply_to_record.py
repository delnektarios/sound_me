import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import librosa
import librosa.display
from scipy.signal import firwin, freqz, lfilter, resample_poly
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')

SAMPLE_RATE  = 44100
MAX_DURATION = 30


# ════════════════════════════════════════════════════════════════════════════
#  RECORDING
# ════════════════════════════════════════════════════════════════════════════

def record_voice(duration=MAX_DURATION, sr=SAMPLE_RATE):
    """Record from the laptop microphone. Press Ctrl+C to stop early."""
    print(f"\nRecording for up to {duration} seconds...")
    print("Speak into your microphone. Press Ctrl+C to stop early.\n")
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr,
                       channels=1, dtype='float32')
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("\nRecording stopped early.")

    audio = audio.flatten()
    audio, _ = librosa.effects.trim(audio, top_db=20)
    actual_duration = len(audio) / sr
    print(f"Recorded {actual_duration:.2f} seconds.")
    sf.write('voice_original.wav', audio, sr)
    print("Saved: voice_original.wav\n")
    return audio, sr


# ════════════════════════════════════════════════════════════════════════════
#  1. WAVEFORM
# ════════════════════════════════════════════════════════════════════════════

def plot_waveform(y, sr):
    time = np.arange(len(y)) / sr
    plt.figure(figsize=(12, 3))
    plt.plot(time, y, color='#4a90d9', lw=0.5)
    plt.title('Your voice — time domain waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  2. FOURIER TRANSFORM
# ════════════════════════════════════════════════════════════════════════════

def plot_fft(y, sr, title='Fourier transform'):
    fft_result = np.fft.rfft(y)
    freqs      = np.fft.rfftfreq(len(y), 1 / sr)
    magnitude  = np.abs(fft_result)

    plt.figure(figsize=(12, 4))
    plt.plot(freqs, magnitude, color='#e74c3c', lw=0.8)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0, 8000)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  3. SPECTROGRAM
# ════════════════════════════════════════════════════════════════════════════

def plot_spectrogram(y, sr, title='Spectrogram (STFT)'):
    D    = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle('Your voice — from waveform to spectrogram', fontsize=13)

    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#4a90d9')
    axes[0].set_title('Time domain waveform')
    axes[0].set_ylabel('Amplitude')

    img = librosa.display.specshow(S_db, sr=sr, x_axis='time',
                                   y_axis='log', ax=axes[1], cmap='magma')
    axes[1].set_title('Spectrogram — frequency content over time')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  4. NYQUIST — OVERSAMPLE / CORRECT / UNDERSAMPLE
# ════════════════════════════════════════════════════════════════════════════

def downsample_no_filter(signal, orig_sr, target_sr):
    """
    Naive integer decimation with NO anti-aliasing filter.
    Frequencies above the Nyquist limit fold back — this IS the aliasing.
    """
    factor = orig_sr // target_sr
    return signal[::factor]


def resample_proper(signal, orig_sr, target_sr):
    """
    Proper resampling WITH anti-aliasing low-pass filter (librosa default).
    Frequencies above the new Nyquist are removed before decimation.
    """
    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr)


def plot_nyquist_on_voice(y, sr):
    """
    Shows spectrograms for oversampled, correctly sampled, and undersampled
    versions of your voice. Undersampling is done WITHOUT an anti-aliasing
    filter so students can both see and hear the aliasing distortion.
    """
    scenarios = [
        ('Oversampling — 96 kHz  (Nyquist = 48 kHz)',  96000, 'proper', '#2ecc71', 'OVER'),
        ('Correct — 16 kHz  (Nyquist = 8 kHz)',         16000, 'proper', '#3498db', 'OK'),
        ('Mild under — 8 kHz, no AA  (Nyquist = 4 kHz)', 8000, 'naive',  '#e67e22', 'UNDER'),
        ('Severe under — 4 kHz, no AA  (Nyquist = 2 kHz)',4000,'naive',  '#e74c3c', 'UNDER'),
        ('Extreme under — 2 kHz, no AA  (Nyquist = 1 kHz)',2000,'naive', '#c0392b', 'UNDER'),
    ]

    fig, axes = plt.subplots(len(scenarios) + 1, 1,
                             figsize=(14, 3 * (len(scenarios) + 1)))
    fig.suptitle('Nyquist on your voice — over / correct / under sampling\n'
                 'Listen to each saved .wav to hear the difference',
                 fontsize=13, fontweight='bold')

    # Original reference row
    D_orig = librosa.stft(y)
    S_orig = librosa.amplitude_to_db(np.abs(D_orig), ref=np.max)
    librosa.display.specshow(S_orig, sr=sr, x_axis='time',
                             y_axis='hz', ax=axes[0], cmap='magma')
    axes[0].set_title(f'Original — {sr // 1000} kHz  (Nyquist = {sr // 2000} kHz)',
                      fontweight='bold')
    axes[0].set_ylim(0, 8000)
    axes[0].set_ylabel('Hz')

    for i, (label, target_sr, method, color, status) in enumerate(scenarios, start=1):
        if method == 'naive':
            factor      = sr // target_sr
            y_decimated = downsample_no_filter(y, sr, target_sr)
            y_playback  = resample_poly(y_decimated, factor, 1)
            if len(y_playback) > len(y):
                y_playback = y_playback[:len(y)]
            else:
                y_playback = np.pad(y_playback, (0, len(y) - len(y_playback)))
            fname = f'voice_UNDER_{target_sr}hz.wav'
            sf.write(fname, y_playback.astype(np.float32), sr)
            D = librosa.stft(y_decimated)
            S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(S, sr=target_sr, x_axis='time',
                                     y_axis='hz', ax=axes[i], cmap='hot')
        else:
            y_resampled = resample_proper(y, sr, target_sr)
            fname       = f'voice_OK_{target_sr}hz.wav'
            sf.write(fname, y_resampled.astype(np.float32), target_sr)
            D = librosa.stft(y_resampled)
            S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(S, sr=target_sr, x_axis='time',
                                     y_axis='hz', ax=axes[i], cmap='magma')

        nyq = target_sr / 2
        axes[i].axhline(y=nyq, color='cyan', linestyle='--', lw=1.2,
                        label=f'Nyquist = {int(nyq)} Hz')
        axes[i].set_title(label, color=color, fontsize=10)
        axes[i].set_ylabel('Hz')
        axes[i].set_ylim(0, 8000)
        axes[i].legend(fontsize=8, loc='upper right')
        print(f'  Saved: {fname}')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_aliasing_detail(y, sr):
    fig, axes = plt.subplots(4, 1, figsize=(13, 10))
    fig.suptitle('Aliasing in the time domain — waveform distortion\n'
                 '(first 0.5 s of your voice)',
                 fontsize=13, fontweight='bold')

    seg_len      = int(0.5 * sr)
    y_seg        = y[:seg_len]
    t_orig       = np.arange(seg_len) / sr
    target_rates = [16000, 8000, 4000]
    colors       = ['#3498db', '#e67e22', '#e74c3c']
    labels       = ['16 kHz (correct — clean)',
                    '8 kHz (mild aliasing)',
                    '4 kHz (severe aliasing)']

    axes[0].plot(t_orig, y_seg, color='#4a90d9', lw=0.8)
    axes[0].set_title(f'Original — {sr // 1000} kHz')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    for i, (target_sr, col, lbl) in enumerate(
            zip(target_rates, colors, labels), start=1):
        factor = sr // target_sr
        y_dec  = y_seg[::factor]
        y_up   = resample_poly(y_dec, factor, 1)[:seg_len]
        t_dec  = np.arange(len(y_dec)) / target_sr

        axes[i].plot(t_orig, y_seg, color='#4a90d9', alpha=0.3,
                     lw=0.8, label='Original')
        axes[i].plot(t_orig, y_up, color=col, lw=0.8,
                     label='Aliased (upsampled back)')

        # Draw sample markers manually — stem() doesn't accept hex colors
        axes[i].vlines(t_dec, 0, y_dec, color=col, alpha=0.4,
                       linewidth=0.8, linestyle='--')
        axes[i].scatter(t_dec, y_dec, color=col, s=8, zorder=3,
                        label=f'Samples at {target_sr} Hz')

        axes[i].set_title(f'{lbl}  —  Nyquist = {target_sr // 2} Hz', color=col)
        axes[i].set_ylabel('Amplitude')
        axes[i].legend(fontsize=8, loc='upper right')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  5. FILTERS
# ════════════════════════════════════════════════════════════════════════════

def apply_and_plot_filters(y, sr):
    """
    Applies low-pass, high-pass, band-pass and band-stop (notch) filters
    to your voice. Each result is saved as a .wav so students can listen.
    Left column = time domain, right column = frequency domain.
    """
    nyquist = sr / 2

    filter_configs = [
        ('Low-pass  (keeps bass / warmth, cutoff 1500 Hz)',
         firwin(1001, 1500 / nyquist),
         '#3498db', 'voice_lowpass.wav'),
        ('High-pass  (removes bass, cutoff 300 Hz)',
         firwin(1001, 300 / nyquist, pass_zero=False),
         '#2ecc71', 'voice_highpass.wav'),
        ('Band-pass  (telephone range 300–3400 Hz)',
         firwin(1001, [300 / nyquist, 3400 / nyquist], pass_zero=False),
         '#e67e22', 'voice_bandpass.wav'),
        ('Band-stop / notch  (removes 900–1100 Hz)',
         firwin(1001, [900 / nyquist, 1100 / nyquist]),
         '#e74c3c', 'voice_notch.wav'),
    ]

    fig, axes = plt.subplots(len(filter_configs) + 1, 2,
                             figsize=(14, 3 * (len(filter_configs) + 1)))
    fig.suptitle('Filters applied to your voice', fontsize=13, fontweight='bold')

    freqs    = np.fft.rfftfreq(len(y), 1 / sr)
    fft_orig = np.abs(np.fft.rfft(y))
    time     = np.arange(len(y)) / sr

    # Original row
    axes[0, 0].plot(time, y, color='#4a90d9', lw=0.5)
    axes[0, 0].set_title('Original')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs, fft_orig, color='#4a90d9', lw=0.8)
    axes[0, 1].set_title('Original spectrum')
    axes[0, 1].set_xlim(0, 6000)
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)

    for row, (name, h, color, fname) in enumerate(filter_configs, start=1):
        y_filtered = lfilter(h, 1.0, y)
        fft_filt   = np.abs(np.fft.rfft(y_filtered))
        sf.write(fname, y_filtered.astype(np.float32), sr)

        axes[row, 0].plot(time, y,          color='#4a90d9', alpha=0.3, lw=0.5)
        axes[row, 0].plot(time, y_filtered, color=color,     lw=0.5)
        axes[row, 0].set_title(name)
        axes[row, 0].set_ylabel('Amplitude')
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(freqs, fft_orig, color='#4a90d9', alpha=0.4,
                          lw=0.8, label='Original')
        axes[row, 1].plot(freqs, fft_filt, color=color,
                          lw=0.8, label='Filtered')
        axes[row, 1].set_xlim(0, 6000)
        axes[row, 1].set_ylabel('Amplitude')
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)
        print(f'  Saved: {fname}')

    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


def plot_filter_frequency_responses(sr):
    """
    Shows the frequency response (magnitude in dB) of each filter kernel
    so students can see exactly which frequencies are passed or blocked.
    """
    nyquist = sr / 2

    filters = [
        ('Low-pass  1500 Hz',
         firwin(1001, 1500 / nyquist), '#3498db'),
        ('High-pass  300 Hz',
         firwin(1001, 300 / nyquist, pass_zero=False), '#2ecc71'),
        ('Band-pass  300–3400 Hz',
         firwin(1001, [300 / nyquist, 3400 / nyquist], pass_zero=False), '#e67e22'),
        ('Band-stop  900–1100 Hz',
         firwin(1001, [900 / nyquist, 1100 / nyquist]), '#e74c3c'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Filter frequency responses — what each filter passes or blocks',
                 fontsize=13, fontweight='bold')

    for ax, (name, h, color) in zip(axes.flat, filters):
        w, H = freqz(h, worN=8000, fs=sr)
        ax.plot(w, 20 * np.log10(np.abs(H) + 1e-10), color=color, lw=1.5)
        ax.set_title(name)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_xlim(0, 6000)
        ax.set_ylim(-80, 5)
        ax.axhline(y=-3, color='gray', linestyle='--', lw=1,
                   label='−3 dB cutoff')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  6. PITCH SHIFT
# ════════════════════════════════════════════════════════════════════════════

def plot_pitch_shift(y, sr):
    """
    Shifts your voice up and down in semitones using librosa's phase vocoder.
    Spectrograms show the formant shift; saved .wav files let students hear it.
    """
    shifts = [-6, -3, 0, 3, 6]
    labels = ['-6 semitones (lower)', '-3 semitones', 'Original',
              '+3 semitones', '+6 semitones (higher)']
    colors = ['#3498db', '#5dade2', '#2ecc71', '#e67e22', '#e74c3c']

    fig, axes = plt.subplots(len(shifts), 1, figsize=(13, 12))
    fig.suptitle('Pitch shifting on your voice — using the phase vocoder',
                 fontsize=13, fontweight='bold')

    for i, (n_steps, lbl, col) in enumerate(zip(shifts, labels, colors)):
        if n_steps == 0:
            y_shifted = y
        else:
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            sf.write(f'voice_pitch_{n_steps:+d}.wav', y_shifted, sr)

        D    = librosa.stft(y_shifted)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time',
                                 y_axis='log', ax=axes[i], cmap='magma')
        axes[i].set_title(lbl, color=col)
        axes[i].set_ylabel('Hz')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    print("Saved: voice_pitch_-6.wav, voice_pitch_-3.wav, "
          "voice_pitch_+3.wav, voice_pitch_+6.wav")


# ════════════════════════════════════════════════════════════════════════════
#  7. FREQUENCY DOMAIN AMPLITUDE SCALING
# ════════════════════════════════════════════════════════════════════════════

def plot_frequency_domain_changes(y, sr):
    """
    Scales amplitude directly in the frequency domain via the FFT,
    then reconstructs with IFFT. Shows both the spectrum and the
    resulting waveform side by side.
    """
    scales = [0.25, 0.5, 1.0, 1.5, 2.0]
    colors = ['#e74c3c', '#e67e22', '#4a90d9', '#2ecc71', '#9b59b6']

    freqs    = np.fft.rfftfreq(len(y), 1 / sr)
    fft_orig = np.fft.rfft(y)
    time     = np.arange(len(y)) / sr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Amplitude scaling in the frequency domain', fontsize=13)

    for scale, col in zip(scales, colors):
        mag = np.abs(fft_orig * scale)
        ax1.plot(freqs, mag, color=col, lw=0.8, alpha=0.8, label=f'×{scale}')

    ax1.set_title('Frequency domain — scaled spectra')
    ax1.set_xlim(0, 6000)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for scale, col in zip(scales, colors):
        y_scaled = np.fft.irfft(fft_orig * scale, n=len(y))
        ax2.plot(time[:sr // 4], y_scaled[:sr // 4],
                 color=col, lw=0.6, alpha=0.7, label=f'×{scale}')

    ax2.set_title('Time domain — first 0.25 s')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Record ──────────────────────────────────────────────────────────────
    y, sr = record_voice(duration=MAX_DURATION, sr=SAMPLE_RATE)

    # ── 1. Waveform ─────────────────────────────────────────────────────────
    print("\n=== 1. Time-domain waveform ===")
    plot_waveform(y, sr)

    # ── 2. FFT ──────────────────────────────────────────────────────────────
    print("\n=== 2. Fourier transform ===")
    plot_fft(y, sr)

    # ── 3. Spectrogram ──────────────────────────────────────────────────────
    print("\n=== 3. Spectrogram ===")
    plot_spectrogram(y, sr)

    # ── 4. Nyquist ──────────────────────────────────────────────────────────
    print("\n=== 4. Nyquist — over / correct / undersampling (spectrograms) ===")
    plot_nyquist_on_voice(y, sr)

    print("\n=== 4b. Nyquist — aliasing in the time domain (waveform zoom) ===")
    plot_aliasing_detail(y, sr)

    # ── 5. Filters ──────────────────────────────────────────────────────────
    print("\n=== 5a. Filters applied to your voice ===")
    apply_and_plot_filters(y, sr)

    print("\n=== 5b. Filter frequency responses ===")
    plot_filter_frequency_responses(sr)

    # ── 6. Pitch shift ──────────────────────────────────────────────────────
    print("\n=== 6. Pitch shifting ===")
    plot_pitch_shift(y, sr)

    # ── 7. Frequency domain amplitude scaling ───────────────────────────────
    print("\n=== 7. Frequency domain amplitude scaling ===")
    plot_frequency_domain_changes(y, sr)

    print("\n=== All done. Files saved in your working directory: ===")
    print("  voice_original.wav")
    print("  voice_OK_96000hz.wav, voice_OK_16000hz.wav")
    print("  voice_UNDER_8000hz.wav, voice_UNDER_4000hz.wav, voice_UNDER_2000hz.wav")
    print("  voice_lowpass.wav, voice_highpass.wav, voice_bandpass.wav, voice_notch.wav")
    print("  voice_pitch_-6.wav, voice_pitch_-3.wav, voice_pitch_+3.wav, voice_pitch_+6.wav")