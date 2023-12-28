import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

plt.style.use('default')


def plot_time_domain_waveform(y, sr, title='Time Domain Waveform'):
    # Calculate time values
    time = np.arange(0, len(y)) / sr

    # Plot the time-domain waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, y)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def plot_fourier_transform(y, sr, title='Fourier Transform'):
    # Compute the Fourier transform
    fft_result = np.fft.fft(y)

    # Calculate the frequencies corresponding to the Fourier transform
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sr)

    # Plot the Fourier transform
    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, np.abs(fft_result))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def apply_frequency_domain_changes(y, sr, amplitude_scale=1.0, frequency_shift=0.0):
    # Compute the Fourier transform
    fft_result = np.fft.fft(y)

    # Apply amplitude scaling
    fft_result *= amplitude_scale

    # Apply frequency shifting
    num_samples = len(fft_result)
    frequencies = np.fft.fftfreq(num_samples, 1 / sr)
    fft_result_shifted = np.fft.fftshift(fft_result)
    frequencies_shifted = np.fft.fftshift(frequencies)
    fft_result_shifted *= np.exp(1j * 2 * np.pi * frequency_shift * frequencies_shifted)
    fft_result = np.fft.ifftshift(fft_result_shifted)

    # Transform back to time domain
    y_modified = np.fft.ifft(fft_result).real

    return y_modified


def remove_bass(y, sr, cutoff_frequency=150.0):
    # Compute the Fourier transform
    fft_result = np.fft.fft(y)

    # Calculate the frequencies corresponding to the Fourier transform
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sr)

    # Apply high-pass filter (remove frequencies below the cutoff)
    fft_result[frequencies < cutoff_frequency] = 0.0

    # Transform back to time domain
    y_filtered = np.fft.ifft(fft_result).real

    return y_filtered


def remove_upper_frequencies(y, sr, cutoff_frequency=5000.0):
    # Compute the Fourier transform
    fft_result = np.fft.fft(y)

    # Calculate the frequencies corresponding to the Fourier transform
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sr)

    # Apply low-pass filter (remove frequencies above the cutoff)
    fft_result[frequencies > cutoff_frequency] = 0.0

    # Transform back to time domain
    y_filtered = np.fft.ifft(fft_result).real

    return y_filtered


def plot_difference(y_original, y_modified, title='Difference Plot'):
    # Plot the difference between two signals
    plt.figure(figsize=(12, 4))
    plt.plot(y_original - y_modified)
    plt.title(title)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude Difference')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Load an MP3 file
    file_path = 'songs/havana.mp3'  # Replace 'your_file.mp3' with the path to your MP3 file
    y, sr = librosa.load(file_path, sr=None)

    # Plot the time-domain waveform
    plot_time_domain_waveform(y, sr, title='Time Domain Waveform')

    # Plot the Fourier transform
    plot_fourier_transform(y, sr)

    # Apply changes to the frequency domain
    y_modified = apply_frequency_domain_changes(y, sr, amplitude_scale=1.5, frequency_shift=100)

    # Save the modified audio as a new WAV file using soundfile
    output_modified_path = 'modified_audio.wav'
    sf.write(output_modified_path, y_modified, sr)

    # Plot the Fourier transform for the original and modified audio
    plot_fourier_transform(y, sr, title='Original Audio Fourier Transform')
    plot_fourier_transform(y_modified, sr, title='Modified Audio Fourier Transform')

    # Remove bass frequencies
    cutoff_frequency = 300.0 # in Hz
    y_filtered = remove_bass(y, sr, cutoff_frequency)

    # Save the filtered audio as a new WAV file using soundfile
    output_filtered_path = 'filtered_audio.wav'
    sf.write(output_filtered_path, y_filtered, sr)

    # Plot the Fourier transform for the original and modified audio
    plot_fourier_transform(y, sr, title='Original Audio Fourier Transform')
    plot_fourier_transform(y_filtered, sr, title='Bass Rm Audio Fourier Transform')

    # Remove upper frequencies
    cutoff_frequency_upper = 5000.0
    y_filtered_upper = remove_upper_frequencies(y, sr, cutoff_frequency_upper)

    # Save the filtered audio as a new WAV file using soundfile
    output_filtered_upper_path = 'filtered_audio_upper.wav'
    sf.write(output_filtered_upper_path, y_filtered_upper, sr)
    # Plot the Fourier transform for the original and modified audio
    plot_fourier_transform(y, sr, title='Original Audio Fourier Transform')
    plot_fourier_transform(y_filtered_upper, sr, title='Upper Audio Fourier Transform')

    plt.tight_layout()
    plt.show()

    # Print the initial song's vector
    print("Initial Song's Vector:")
    print(y)
    # Demonstrate the relationship between sampling rate and the number of values in the vector
    print(f"Sampling Rate: {sr} Hz")
    print(f"Number of Values in the Vector: {len(y)}")
    print(f"Duration of the Audio: {len(y)/sr} seconds")
