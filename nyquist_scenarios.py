import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample_poly

def demo_nyquist_aliasing(sr=44100, duration=3.0):
    """
    Synthesizes a signal with multiple known frequency components,
    then demonstrates what happens when sampled at various rates —
    some above and some below the Nyquist limit for those frequencies.

    Three scenarios are shown and saved as .wav files:

      A) Pure tone aliasing     — a single high-frequency sine,
                                  showing how it folds to a wrong pitch
      B) Sweep aliasing         — a sine that sweeps from low to high,
                                  so students hear the pitch suddenly
                                  distort when crossing the Nyquist limit
      C) Multi-tone aliasing    — several tones, some of which alias
                                  onto each other, causing dissonance
    """

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)


    # ════════════════════════════════════════════════════════════════════════
    # SCENARIO A — Pure tone: 3000 Hz sampled at 4000 Hz
    # Nyquist limit = 2000 Hz → 3000 Hz aliases to |3000 - 4000| = 1000 Hz
    # Students hear a 1000 Hz tone instead of 3000 Hz
    # ════════════════════════════════════════════════════════════════════════

    def synthesize_and_alias(true_freq, sample_rate, orig_sr=sr):
        """
        Generate a sine at true_freq, then simulate sampling at sample_rate
        WITHOUT anti-aliasing. Upsample back for playback at orig_sr.
        """
        signal    = np.sin(2 * np.pi * true_freq * t)
        factor    = orig_sr // sample_rate
        decimated = signal[::factor]
        playback  = resample_poly(decimated, factor, 1)
        if len(playback) > len(signal):
            playback = playback[:len(signal)]
        else:
            playback = np.pad(playback, (0, len(signal) - len(playback)))
        return signal, decimated, playback

    fig_a, axes_a = plt.subplots(3, 2, figsize=(14, 10))
    fig_a.suptitle(
        'Scenario A — Pure tone aliasing\n'
        'True signal: 3000 Hz  |  Sample rate: 4000 Hz  |  '
        'Nyquist: 2000 Hz  →  alias at 1000 Hz',
        fontsize=12, fontweight='bold')

    true_freq   = 3000
    sample_rate = 4000
    alias_freq  = abs(true_freq - sample_rate)   # = 1000 Hz
    factor_a    = sr // sample_rate

    signal_a, dec_a, playback_a = synthesize_and_alias(true_freq, sample_rate)

    # Also generate the alias tone directly so students can compare
    alias_tone  = np.sin(2 * np.pi * alias_freq * t)

    # Time domain — full signal
    zoom = slice(0, sr // 100)   # first 10 ms
    t_zoom = t[zoom]
    t_dec_a = np.arange(len(dec_a)) / sample_rate

    axes_a[0, 0].plot(t_zoom, signal_a[zoom],   color='#3498db', lw=1.5,
                      label=f'True signal ({true_freq} Hz)')
    axes_a[0, 0].plot(t_zoom, alias_tone[zoom],  color='#e74c3c', lw=1.5,
                      linestyle='--', label=f'Alias tone ({alias_freq} Hz)')
    axes_a[0, 0].set_title('True signal vs alias tone (time domain, 10 ms zoom)')
    axes_a[0, 0].set_xlabel('Time (s)')
    axes_a[0, 0].set_ylabel('Amplitude')
    axes_a[0, 0].legend(fontsize=9)
    axes_a[0, 0].grid(True, alpha=0.3)

    # Sample points overlaid
    t_dec_zoom = t_dec_a[t_dec_a < t_zoom[-1]]
    dec_zoom   = dec_a[:len(t_dec_zoom)]
    axes_a[1, 0].plot(t_zoom, signal_a[zoom], color='#3498db', alpha=0.4,
                      lw=1.5, label=f'True signal ({true_freq} Hz)')
    axes_a[1, 0].plot(t_zoom, alias_tone[zoom], color='#e74c3c', lw=1.5,
                      linestyle='--', label=f'What sampler "sees" ({alias_freq} Hz)')
    axes_a[1, 0].vlines(t_dec_zoom, 0, dec_zoom,
                        color='#2ecc71', linewidth=1.2, alpha=0.7)
    axes_a[1, 0].scatter(t_dec_zoom, dec_zoom, color='#2ecc71', s=40,
                         zorder=5, label=f'Sample points ({sample_rate} Hz)')
    axes_a[1, 0].set_title('Sample points connecting to the alias — not the true signal')
    axes_a[1, 0].set_xlabel('Time (s)')
    axes_a[1, 0].set_ylabel('Amplitude')
    axes_a[1, 0].legend(fontsize=9)
    axes_a[1, 0].grid(True, alpha=0.3)

    # Frequency domain
    freqs    = np.fft.rfftfreq(len(signal_a), 1 / sr)
    freqs_d  = np.fft.rfftfreq(len(dec_a),    1 / sample_rate)
    fft_true = np.abs(np.fft.rfft(signal_a))
    fft_dec  = np.abs(np.fft.rfft(dec_a))
    fft_play = np.abs(np.fft.rfft(playback_a))

    axes_a[0, 1].plot(freqs, fft_true, color='#3498db', lw=1.5,
                      label=f'True signal ({true_freq} Hz)')
    axes_a[0, 1].axvline(x=sample_rate / 2, color='red', linestyle='--',
                         lw=1.2, label=f'Nyquist limit ({sample_rate // 2} Hz)')
    axes_a[0, 1].set_title('True signal spectrum')
    axes_a[0, 1].set_xlim(0, 5000)
    axes_a[0, 1].set_xlabel('Frequency (Hz)')
    axes_a[0, 1].set_ylabel('Amplitude')
    axes_a[0, 1].legend(fontsize=9)
    axes_a[0, 1].grid(True, alpha=0.3)

    axes_a[1, 1].plot(freqs_d, fft_dec, color='#e74c3c', lw=1.5,
                      label=f'Sampled at {sample_rate} Hz → alias at {alias_freq} Hz')
    axes_a[1, 1].axvline(x=sample_rate / 2, color='red', linestyle='--',
                         lw=1.2, label=f'Nyquist limit ({sample_rate // 2} Hz)')
    axes_a[1, 1].set_title(f'Sampled spectrum — peak has MOVED to {alias_freq} Hz')
    axes_a[1, 1].set_xlim(0, 5000)
    axes_a[1, 1].set_xlabel('Frequency (Hz)')
    axes_a[1, 1].set_ylabel('Amplitude')
    axes_a[1, 1].legend(fontsize=9)
    axes_a[1, 1].grid(True, alpha=0.3)

    # Alias formula annotation
    axes_a[2, 0].axis('off')
    axes_a[2, 0].text(0.5, 0.6,
        'Alias frequency formula:',
        ha='center', fontsize=13, fontweight='bold',
        transform=axes_a[2, 0].transAxes)
    axes_a[2, 0].text(0.5, 0.35,
        r'$f_{alias} = |f_{signal} - f_{sample}|$',
        ha='center', fontsize=16,
        transform=axes_a[2, 0].transAxes)
    axes_a[2, 0].text(0.5, 0.1,
        f'|{true_freq} − {sample_rate}| = {alias_freq} Hz',
        ha='center', fontsize=13, color='#e74c3c',
        transform=axes_a[2, 0].transAxes)

    # Playback spectrum
    axes_a[2, 1].plot(freqs, fft_play, color='#9b59b6', lw=1.5,
                      label=f'Upsampled playback — alias at {alias_freq} Hz')
    axes_a[2, 1].plot(freqs, fft_true, color='#3498db', lw=1.0,
                      alpha=0.4, label=f'True signal ({true_freq} Hz)')
    axes_a[2, 1].axvline(x=sample_rate / 2, color='red', linestyle='--',
                         lw=1.2, label=f'Nyquist limit ({sample_rate // 2} Hz)')
    axes_a[2, 1].set_title('Playback spectrum — you hear the alias, not the true tone')
    axes_a[2, 1].set_xlim(0, 5000)
    axes_a[2, 1].set_xlabel('Frequency (Hz)')
    axes_a[2, 1].set_ylabel('Amplitude')
    axes_a[2, 1].legend(fontsize=9)
    axes_a[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Save scenario A files
    sf.write('alias_A_true_3000hz.wav',    signal_a.astype(np.float32),    sr)
    sf.write('alias_A_aliased_1000hz.wav', playback_a.astype(np.float32),  sr)
    sf.write('alias_A_expected_1000hz.wav',alias_tone.astype(np.float32),  sr)
    print("Scenario A saved:")
    print("  alias_A_true_3000hz.wav    — what it SHOULD sound like (3000 Hz)")
    print("  alias_A_aliased_1000hz.wav — what the sampler PRODUCES (1000 Hz alias)")
    print("  alias_A_expected_1000hz.wav— pure 1000 Hz tone for comparison")


    # ════════════════════════════════════════════════════════════════════════
    # SCENARIO B — Frequency sweep crossing the Nyquist boundary
    # Sweep from 200 Hz to 6000 Hz. At sample_rate=8000 Hz the Nyquist is
    # 4000 Hz — above that the sweep starts folding back down.
    # Students hear the pitch rise, then suddenly reverse direction.
    # ════════════════════════════════════════════════════════════════════════

    sample_rate_b = 8000
    nyquist_b     = sample_rate_b / 2    # 4000 Hz
    f_start, f_end = 200, 6000

    # Chirp (linear frequency sweep)
    sweep = np.sin(2 * np.pi * (f_start + (f_end - f_start) / (2 * duration) * t) * t)

    factor_b  = sr // sample_rate_b
    dec_b     = sweep[::factor_b]
    play_b    = resample_poly(dec_b, factor_b, 1)
    if len(play_b) > len(sweep):
        play_b = play_b[:len(sweep)]
    else:
        play_b = np.pad(play_b, (0, len(sweep) - len(play_b)))

    fig_b, axes_b = plt.subplots(2, 1, figsize=(14, 7))
    fig_b.suptitle(
        f'Scenario B — Sweep crossing the Nyquist boundary\n'
        f'Sweep: {f_start}→{f_end} Hz  |  Sample rate: {sample_rate_b} Hz  |  '
        f'Nyquist: {int(nyquist_b)} Hz  →  pitch REVERSES above {int(nyquist_b)} Hz',
        fontsize=12, fontweight='bold')

    # Spectrogram of original sweep
    from scipy.signal import spectrogram as scipy_spec
    f_s, t_s, Sxx = scipy_spec(sweep, fs=sr, nperseg=1024, noverlap=512)
    axes_b[0].pcolormesh(t_s, f_s, 10 * np.log10(Sxx + 1e-10),
                         shading='gouraud', cmap='magma')
    axes_b[0].axhline(y=nyquist_b, color='cyan', linestyle='--', lw=1.5,
                      label=f'Nyquist limit ({int(nyquist_b)} Hz)')
    axes_b[0].set_title('Original sweep — rising pitch all the way to 6000 Hz')
    axes_b[0].set_ylabel('Frequency (Hz)')
    axes_b[0].set_ylim(0, 7000)
    axes_b[0].legend(fontsize=9)

    # Spectrogram of aliased sweep — pitch reverses after Nyquist
    f_p, t_p, Sxx_p = scipy_spec(play_b, fs=sr, nperseg=1024, noverlap=512)
    axes_b[1].pcolormesh(t_p, f_p, 10 * np.log10(Sxx_p + 1e-10),
                         shading='gouraud', cmap='hot')
    axes_b[1].axhline(y=nyquist_b, color='cyan', linestyle='--', lw=1.5,
                      label=f'Nyquist limit ({int(nyquist_b)} Hz)')
    axes_b[1].set_title(f'Undersampled at {sample_rate_b} Hz — '
                        f'pitch REVERSES after {int(nyquist_b)} Hz (aliasing)')
    axes_b[1].set_ylabel('Frequency (Hz)')
    axes_b[1].set_xlabel('Time (s)')
    axes_b[1].set_ylim(0, 7000)
    axes_b[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    sf.write('alias_B_sweep_original.wav', sweep.astype(np.float32),  sr)
    sf.write('alias_B_sweep_aliased.wav',  play_b.astype(np.float32), sr)
    print("\nScenario B saved:")
    print("  alias_B_sweep_original.wav — rising sweep, sounds normal")
    print("  alias_B_sweep_aliased.wav  — pitch reverses at Nyquist (very audible!)")


    # ════════════════════════════════════════════════════════════════════════
    # SCENARIO C — Multi-tone: several frequencies, some aliasing onto others
    # Tones: 500, 1500, 3500, 5500 Hz sampled at 6000 Hz
    # Nyquist = 3000 Hz:
    #   500  Hz → stays at  500 Hz  (below Nyquist — safe)
    #   1500 Hz → stays at 1500 Hz  (below Nyquist — safe)
    #   3500 Hz → aliases to |3500 - 6000| = 2500 Hz
    #   5500 Hz → aliases to |5500 - 6000| =  500 Hz  ← collides with real 500 Hz tone!
    # ════════════════════════════════════════════════════════════════════════

    sample_rate_c = 6000
    nyquist_c     = sample_rate_c / 2    # 3000 Hz
    tones         = [500, 1500, 3500, 5500]
    aliases       = [abs(f - sample_rate_c) if f > nyquist_c else f for f in tones]

    multi = sum(np.sin(2 * np.pi * f * t) for f in tones) / len(tones)
    factor_c = sr // sample_rate_c
    dec_c    = multi[::factor_c]
    play_c   = resample_poly(dec_c, factor_c, 1)
    if len(play_c) > len(multi):
        play_c = play_c[:len(multi)]
    else:
        play_c = np.pad(play_c, (0, len(multi) - len(play_c)))

    fig_c, axes_c = plt.subplots(2, 1, figsize=(14, 8))
    fig_c.suptitle(
        f'Scenario C — Multi-tone aliasing and frequency collision\n'
        f'Tones: {tones} Hz  |  Sample rate: {sample_rate_c} Hz  |  '
        f'Nyquist: {int(nyquist_c)} Hz',
        fontsize=12, fontweight='bold')

    freqs_c  = np.fft.rfftfreq(len(multi), 1 / sr)
    freqs_dc = np.fft.rfftfreq(len(dec_c),  1 / sample_rate_c)
    fft_c    = np.abs(np.fft.rfft(multi))
    fft_dc   = np.abs(np.fft.rfft(dec_c))

    axes_c[0].plot(freqs_c, fft_c, color='#3498db', lw=1.5)
    for f in tones:
        axes_c[0].axvline(x=f, color='#3498db', linestyle=':', lw=1, alpha=0.6)
        axes_c[0].text(f + 30, axes_c[0].get_ylim()[1] * 0.8,
                       f'{f} Hz', fontsize=9, color='#3498db', rotation=90)
    axes_c[0].axvline(x=nyquist_c, color='red', linestyle='--', lw=1.5,
                      label=f'Nyquist = {int(nyquist_c)} Hz')
    axes_c[0].set_title('Original spectrum — 4 clean tones')
    axes_c[0].set_xlim(0, 7000)
    axes_c[0].set_xlabel('Frequency (Hz)')
    axes_c[0].set_ylabel('Amplitude')
    axes_c[0].legend(fontsize=9)
    axes_c[0].grid(True, alpha=0.3)

    axes_c[1].plot(freqs_dc, fft_dc, color='#e74c3c', lw=1.5)
    for f, a in zip(tones, aliases):
        col   = '#e74c3c' if f > nyquist_c else '#2ecc71'
        label = f'{f}→{a} Hz (ALIAS)' if f > nyquist_c else f'{f} Hz (safe)'
        axes_c[1].axvline(x=a, color=col, linestyle=':', lw=1.2, alpha=0.8)
        axes_c[1].text(a + 30, max(fft_dc) * 0.6,
                       label, fontsize=8, color=col, rotation=90)
    axes_c[1].axvline(x=nyquist_c, color='red', linestyle='--', lw=1.5,
                      label=f'Nyquist = {int(nyquist_c)} Hz')
    axes_c[1].set_title(
        f'Aliased spectrum — 3500→2500 Hz, 5500→500 Hz (collides with real 500 Hz tone!)')
    axes_c[1].set_xlim(0, 4000)
    axes_c[1].set_xlabel('Frequency (Hz)')
    axes_c[1].set_ylabel('Amplitude')
    axes_c[1].legend(fontsize=9)
    axes_c[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    sf.write('alias_C_multitone_original.wav', multi.astype(np.float32),   sr)
    sf.write('alias_C_multitone_aliased.wav',  play_c.astype(np.float32),  sr)
    print("\nScenario C saved:")
    print("  alias_C_multitone_original.wav — 4 clean tones (500, 1500, 3500, 5500 Hz)")
    print("  alias_C_multitone_aliased.wav  — aliased: dissonant, 500 Hz doubled in amplitude")


if __name__ == '__main__':
    demo_nyquist_aliasing()