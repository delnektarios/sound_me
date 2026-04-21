import librosa.display

def plot_spectrogram(y, sr, title='Spectrogram (STFT)'):
    """
    A spectrogram shows how the frequency content of a signal
    changes over time — something a plain FFT cannot show.
    Uses the Short-Time Fourier Transform (STFT).
    """
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('From Waveform to Spectrogram', fontsize=14, fontweight='bold')

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#4a90d9')
    axes[0].set_title('① Time Domain Waveform')
    axes[0].set_ylabel('Amplitude')

    # Static FFT (loses time info)
    fft_result = np.abs(np.fft.rfft(y))
    freqs      = np.fft.rfftfreq(len(y), 1 / sr)
    axes[1].plot(freqs, fft_result, color='#e74c3c')
    axes[1].set_title('② Fourier Transform — frequency content but NO time info')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlim(0, sr / 2)
    axes[1].grid(True, alpha=0.4)

    # Spectrogram
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time',
                                   y_axis='log', ax=axes[2], cmap='magma')
    axes[2].set_title('③ Spectrogram (STFT) — frequency AND time')
    fig.colorbar(img, ax=axes[2], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load an example audio file
    y, sr = librosa.load(librosa.ex('trumpet'))

    # Plot the spectrogram
    plot_spectrogram(y, sr)