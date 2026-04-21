import numpy as np
import matplotlib.pyplot as plt

def plot_real_world_examples():
    """
    Show real-world sampling rates and what Nyquist frequency they imply.
    Helps students connect the theorem to technology they already use.
    """
    examples = [
        ('Phone call\n(8 kHz)',        8000,   4000,  '#e74c3c'),
        ('AM Radio\n(22 kHz)',         22000,  11000, '#e67e22'),
        ('CD Audio\n(44.1 kHz)',       44100,  22050, '#2ecc71'),
        ('DVD Audio\n(48 kHz)',        48000,  24000, '#27ae60'),
        ('Studio Recording\n(96 kHz)', 96000,  48000, '#1abc9c'),
        ('Hi-Res Audio\n(192 kHz)',    192000, 96000, '#16a085'),
    ]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    names      = [e[0] for e in examples]
    samp_rates = [e[1] for e in examples]
    nyq_freqs  = [e[2] for e in examples]
    colors     = [e[3] for e in examples]
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, samp_rates, width, label='Sample Rate (Hz)',
                   color=colors, alpha=0.85)
    bars2 = ax.bar(x + width/2, nyq_freqs,  width, label='Nyquist Frequency (Hz)',
                   color=colors, alpha=0.4)

    # Human hearing limit
    ax.axhline(y=20000, color='red', linestyle='--', linewidth=1.5,
               label='Human hearing limit (20 kHz)')

    # Value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1000,
                f'{h/1000:.1f}k', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1000,
                f'{h/1000:.1f}k', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Real-World Sampling Rates & Their Nyquist Frequencies',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_nyquist_rule_explanation():
    """
    Visual introduction to the Nyquist rule before the demonstrations.
    Shows the theorem statement, the minimum sampling requirement,
    and a simple intuitive illustration.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#f8f9fa')

    # ── Title ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.95, 'The Nyquist–Shannon Sampling Theorem',
             ha='center', va='top', fontsize=20, fontweight='bold', color='#1a1a2e')
    fig.text(0.5, 0.89,
             'To perfectly reconstruct a signal, the sample rate must be\n'
             'at least twice the highest frequency present in the signal.',
             ha='center', va='top', fontsize=13, color='#333333',
             linespacing=1.6)

    # ── Formula box ─────────────────────────────────────────────────────────
    formula_ax = fig.add_axes([0.3, 0.72, 0.4, 0.1])
    formula_ax.set_xlim(0, 1)
    formula_ax.set_ylim(0, 1)
    formula_ax.set_facecolor('#1a1a2e')
    for spine in formula_ax.spines.values():
        spine.set_visible(False)
    formula_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    formula_ax.text(0.5, 0.55, r'$f_{sample} \geq 2 \times f_{max}$',
                    ha='center', va='center', fontsize=18, color='white',
                    fontweight='bold')
    formula_ax.text(0.5, 0.1, r'where  $f_{Nyquist} = f_{sample}\ /\ 2$',
                    ha='center', va='center', fontsize=11, color='#aaaacc')

    # ── Left panel: Nyquist satisfied ───────────────────────────────────────
    ax_good = fig.add_axes([0.05, 0.32, 0.4, 0.32])
    t_cont = np.linspace(0, 1, 1000)
    y_cont = np.sin(2 * np.pi * 3 * t_cont)
    t_samp = np.arange(0, 1, 1 / 20)          # 20 Hz > 2×3 Hz
    y_samp = np.sin(2 * np.pi * 3 * t_samp)

    ax_good.plot(t_cont, y_cont, color='#4a90d9', lw=2, label='True signal (3 Hz)', zorder=2)
    ax_good.stem(t_samp, y_samp, linefmt='g-', markerfmt='go', basefmt='k-',
                 label='Samples (20 Hz)')
    ax_good.set_facecolor('#f0fff0')
    ax_good.set_title('✔  Nyquist Satisfied\n'
                      r'$f_{sample}=20\ Hz \geq 2\times3\ Hz$',
                      color='#2a7a2a', fontsize=11, fontweight='bold')
    ax_good.set_xlabel('Time (s)')
    ax_good.set_ylabel('Amplitude')
    ax_good.legend(fontsize=8)
    ax_good.grid(True, alpha=0.4)
    ax_good.set_ylim(-1.5, 1.8)

    # ── Right panel: Nyquist violated ───────────────────────────────────────
    ax_bad = fig.add_axes([0.55, 0.32, 0.4, 0.32])
    t_samp_bad = np.arange(0, 1, 1 / 5)       # 5 Hz < 2×3 Hz
    y_samp_bad = np.sin(2 * np.pi * 3 * t_samp_bad)
    alias_freq = abs(3 - 5)
    y_alias = np.sin(2 * np.pi * alias_freq * t_cont)

    ax_bad.plot(t_cont, y_cont, color='#4a90d9', lw=2, alpha=0.4, label='True signal (3 Hz)')
    ax_bad.plot(t_cont, y_alias, color='#e07b00', lw=2, linestyle='--',
                label=f'Alias ({alias_freq} Hz)')
    ax_bad.stem(t_samp_bad, y_samp_bad, linefmt='r-', markerfmt='ro', basefmt='k-',
                label='Samples (5 Hz)')
    ax_bad.set_facecolor('#fff0f0')
    ax_bad.set_title('✘  Nyquist Violated\n'
                     r'$f_{sample}=5\ Hz < 2\times3\ Hz$',
                     color='#aa0000', fontsize=11, fontweight='bold')
    ax_bad.set_xlabel('Time (s)')
    ax_bad.set_ylabel('Amplitude')
    ax_bad.legend(fontsize=8)
    ax_bad.grid(True, alpha=0.4)
    ax_bad.set_ylim(-1.5, 1.8)

    # ── Bottom: key terms ───────────────────────────────────────────────────
    terms = [
        ('Sampling Rate  $(f_{sample})$',  'How many times per second\nthe signal is measured'),
        ('Nyquist Frequency  $(f_{sample}/2)$', 'The highest frequency that\ncan be correctly captured'),
        ('Aliasing',                        'Distortion caused when a high-\nfrequency signal is misread\nas a lower frequency'),
    ]

    for i, (term, definition) in enumerate(terms):
        x = 0.1 + i * 0.3
        term_ax = fig.add_axes([x + 0.02, 0.04, 0.24, 0.2])
        term_ax.set_facecolor('#1a1a2e')
        term_ax.set_xlim(0, 1)
        term_ax.set_ylim(0, 1)
        for spine in term_ax.spines.values():
            spine.set_visible(False)
        term_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        term_ax.text(0.5, 0.72, term, ha='center', va='center',
                     fontsize=10, color='#7ec8e3', fontweight='bold')
        term_ax.text(0.5, 0.3, definition, ha='center', va='center',
                     fontsize=9, color='#cccccc', linespacing=1.5)

    plt.show()


def plot_nyquist_proper_sampling(frequency=5.0, sample_rate=100.0, duration=1.0):
    """
    Demonstrate proper sampling: sample_rate > 2 * signal_frequency (Nyquist satisfied).
    """
    # Continuous signal (high resolution simulation)
    t_continuous = np.linspace(0, duration, 10000)
    y_continuous = np.sin(2 * np.pi * frequency * t_continuous)

    # Sampled signal
    t_sampled = np.arange(0, duration, 1 / sample_rate)
    y_sampled = np.sin(2 * np.pi * frequency * t_sampled)

    plt.figure(figsize=(12, 4))
    plt.plot(t_continuous, y_continuous, label='Continuous Signal', color='blue', alpha=0.6)
    plt.stem(t_sampled, y_sampled, linefmt='r-', markerfmt='ro', basefmt='k-',
             label=f'Sampled ({sample_rate} Hz)')
    plt.title(f'Proper Sampling — Signal: {frequency} Hz, Sample Rate: {sample_rate} Hz\n'
              f'Nyquist satisfied: {sample_rate} > {2 * frequency}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_nyquist_aliasing(frequency=45.0, sample_rate=50.0, duration=1.0):
    """
    Demonstrate aliasing: sample_rate < 2 * signal_frequency (Nyquist violated).
    The high-frequency signal is misinterpreted as a much lower frequency.
    """
    # Continuous signal
    t_continuous = np.linspace(0, duration, 10000)
    y_continuous = np.sin(2 * np.pi * frequency * t_continuous)

    # Sampled signal (too few samples)
    t_sampled = np.arange(0, duration, 1 / sample_rate)
    y_sampled = np.sin(2 * np.pi * frequency * t_sampled)

    # The alias frequency the sampler "thinks" it sees
    alias_frequency = abs(frequency - sample_rate)
    y_alias = np.sin(2 * np.pi * alias_frequency * t_continuous)

    plt.figure(figsize=(12, 4))
    plt.plot(t_continuous, y_continuous, label=f'True Signal ({frequency} Hz)',
             color='blue', alpha=0.4)
    plt.plot(t_continuous, y_alias, label=f'Alias Signal ({alias_frequency} Hz)',
             color='green', linestyle='--')
    plt.stem(t_sampled, y_sampled, linefmt='r-', markerfmt='ro', basefmt='k-',
             label=f'Sampled ({sample_rate} Hz)')
    plt.title(f'Aliasing — Signal: {frequency} Hz, Sample Rate: {sample_rate} Hz\n'
              f'Nyquist VIOLATED: {sample_rate} < {2 * frequency} — perceived as {alias_frequency} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_nyquist_frequency_spectrum(frequency=5.0, sample_rate=100.0, duration=1.0):
    """
    Show the Nyquist limit visually in the frequency domain.
    The Nyquist frequency (sample_rate / 2) is the highest frequency
    that can be correctly represented.
    """
    t = np.arange(0, duration, 1 / sample_rate)
    y = np.sin(2 * np.pi * frequency * t)

    # Fourier transform
    fft_result = np.fft.fft(y)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    nyquist = sample_rate / 2

    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, np.abs(fft_result), color='blue', label='Frequency Spectrum')
    plt.axvline(x=nyquist, color='red', linestyle='--', label=f'Nyquist Limit ({nyquist} Hz)')
    plt.axvline(x=-nyquist, color='red', linestyle='--')
    plt.title(f'Frequency Spectrum with Nyquist Limit\n'
              f'Sample Rate: {sample_rate} Hz — Max representable frequency: {nyquist} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_nyquist_comparison(frequency=45.0, sample_rates=[200, 100, 50, 48], duration=0.5):
    """
    Side-by-side comparison of the same signal sampled at different rates,
    showing how dropping below the Nyquist threshold introduces aliasing.
    """
    t_continuous = np.linspace(0, duration, 10000)
    y_continuous = np.sin(2 * np.pi * frequency * t_continuous)

    fig, axes = plt.subplots(len(sample_rates), 1, figsize=(12, 3 * len(sample_rates)))
    fig.suptitle(f'Same Signal ({frequency} Hz) at Different Sample Rates', fontsize=14)

    for ax, sr in zip(axes, sample_rates):
        t_sampled = np.arange(0, duration, 1 / sr)
        y_sampled = np.sin(2 * np.pi * frequency * t_sampled)

        nyquist_ok = sr > 2 * frequency
        color = 'green' if nyquist_ok else 'red'
        status = 'OK' if nyquist_ok else 'ALIASING'

        ax.plot(t_continuous, y_continuous, color='blue', alpha=0.3, label='True signal')
        ax.stem(t_sampled, y_sampled, linefmt=f'{color}-', markerfmt=f'{color}o',
                basefmt='k-', label=f'{sr} Hz [{status}]')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title(f'Sample Rate: {sr} Hz — Nyquist: {sr / 2} Hz — {status}')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_nyquist_rule_explanation()        # 0. Intro figure
    plot_real_world_examples()             # 1. Context — why does this matter?
    plot_nyquist_proper_sampling()         # 2. Good sampling
    plot_nyquist_aliasing()                # 3. Bad sampling
    plot_nyquist_frequency_spectrum()      # 4. Frequency domain view
    plot_nyquist_comparison()              # 5. Side-by-side rates
    # plot_signal_reconstruction()           # 6. Reconstruction — completing the picture
    # interactive_aliasing_explorer()        # 7. Interactive — last, as a playground
    
