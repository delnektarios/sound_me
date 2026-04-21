import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import firwin

def plot_convolution_explanation():
    """
    Step-by-step visual explanation of discrete convolution:
    shows the flip-and-slide mechanism that students often find confusing.
    """
    # Simple signals easy to follow visually
    signal = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    kernel = np.array([0.2, 0.5, 0.2], dtype=float)   # simple smoothing kernel

    result = np.convolve(signal, kernel, mode='full')
    n_sig  = len(signal)
    n_ker  = len(kernel)
    n_res  = len(result)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11))
    fig.suptitle('How Discrete Convolution Works — Flip & Slide',
                 fontsize=15, fontweight='bold')

    # ── ① Original signal ───────────────────────────────────────────────────
    axes[0].bar(range(n_sig), signal, color='#4a90d9', alpha=0.8, label='Signal x[n]')
    axes[0].set_title('① Input Signal  x[n]')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(-1, n_res)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    for i, v in enumerate(signal):
        if v != 0:
            axes[0].text(i, v + 0.05, f'{v:.0f}', ha='center', fontsize=9)

    # ── ② Kernel (and its flipped version) ──────────────────────────────────
    ax2 = axes[1]
    x_ker = np.arange(n_ker)
    ax2.bar(x_ker, kernel, color='#e74c3c', alpha=0.8, label='Kernel h[n]')
    ax2.bar(x_ker + 7, kernel[::-1], color='#e67e22', alpha=0.8,
            label='Flipped kernel h[-n]')
    ax2.annotate('', xy=(7.5, 0.55), xytext=(3.5, 0.55),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax2.text(5.3, 0.57, 'flip', fontsize=10, fontweight='bold')
    ax2.set_title('② The Kernel  h[n]  is Flipped Before Sliding\n'
                  '(for symmetric kernels like this one, flipping changes nothing — '
                  'but in general it does)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(-1, n_res)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    for i, v in enumerate(kernel):
        ax2.text(i, v + 0.02, f'{v}', ha='center', fontsize=9)
    for i, v in enumerate(kernel[::-1]):
        ax2.text(i + 7, v + 0.02, f'{v}', ha='center', fontsize=9, color='#e67e22')

    # ── ③ One step of the slide: show multiply-and-sum at position k=3 ──────
    k = 4   # the position we are illustrating
    ax3 = axes[2]

    # Place flipped kernel aligned at position k
    ker_flipped  = kernel[::-1]
    ker_positions = np.arange(k, k + n_ker)
    products      = np.array([
        signal[p] * ker_flipped[i] if 0 <= p < n_sig else 0
        for i, p in enumerate(ker_positions)
    ])

    ax3.bar(range(n_sig), signal, color='#4a90d9', alpha=0.3, label='Signal x[n]')
    ax3.bar(ker_positions, ker_flipped * 1.2, color='#e67e22', alpha=0.5,
            label=f'Kernel positioned at k={k}', width=0.4)
    ax3.bar(ker_positions - 0.2, products, color='#2ecc71', alpha=0.9,
            label=f'Products (sum = {sum(products):.2f} = output at k={k})',
            width=0.4)

    ax3.axvline(x=k + (n_ker - 1) / 2, color='red', linestyle='--', lw=1.5,
                label=f'Current position k={k}')
    ax3.set_title(f'③ At Each Position k: Multiply Overlapping Values & Sum\n'
                  f'Output[{k}] = {" + ".join([f"{signal[p]:.0f}×{ker_flipped[i]:.1f}" for i, p in enumerate(ker_positions) if 0 <= p < n_sig])} = {result[k]:.2f}')
    ax3.set_ylabel('Amplitude')
    ax3.set_xlim(-1, n_res)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # ── ④ Full convolution result ────────────────────────────────────────────
    axes[3].bar(range(n_sig), signal, color='#4a90d9', alpha=0.4, label='Original signal')
    axes[3].plot(range(n_res), result, color='#e74c3c', lw=2.5, marker='o',
                 markersize=5, label='Convolution result  (x * h)[n]')
    axes[3].set_title('④ Full Convolution Result — the signal has been smoothed')
    axes[3].set_xlabel('Sample index n')
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlim(-1, n_res)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_convolution_interactive():
    """
    Slider-driven demo: watch the kernel slide across the signal
    one step at a time. Students see exactly which values are
    being multiplied and summed at every position.
    """
    signal = np.array([0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    kernel = np.array([0.2, 0.5, 0.2], dtype=float)
    result = np.convolve(signal, kernel, mode='full')

    n_sig = len(signal)
    n_ker = len(kernel)
    n_res = len(result)
    ker_flipped = kernel[::-1]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 7))
    plt.subplots_adjust(bottom=0.18)
    fig.suptitle('Interactive Convolution — Slide the Kernel Step by Step',
                 fontsize=13, fontweight='bold')

    def draw(k):
        ax_top.cla()
        ax_bot.cla()

        # Compute products at this position
        products = []
        prod_positions = []
        for i in range(n_ker):
            p = k - (n_ker - 1) + i        # signal index being touched
            if 0 <= p < n_sig:
                products.append(signal[p] * ker_flipped[i])
                prod_positions.append(p)

        output_val = sum(products)

        # ── Top: signal + sliding kernel ────────────────────────────────────
        ax_top.bar(range(n_sig), signal, color='#4a90d9', alpha=0.5,
                   label='Signal x[n]', zorder=2)

        # Highlight touched samples
        for p in prod_positions:
            ax_top.bar(p, signal[p], color='#4a90d9', alpha=1.0, zorder=3)

        # Draw kernel at current position
        ker_x = [k - (n_ker - 1) + i for i in range(n_ker)]
        ax_top.bar(ker_x, ker_flipped * 3, color='#e67e22', alpha=0.6,
                   width=0.4, label='Kernel h[-n] (scaled for visibility)', zorder=4)

        # Products
        for pos, prod in zip(prod_positions, products):
            ax_top.bar(pos + 0.35, prod, color='#2ecc71', alpha=0.9,
                       width=0.35, zorder=5)

        ax_top.axvline(x=k, color='red', linestyle='--', lw=1.5,
                       label=f'k = {k}  →  output[{k}] = {output_val:.3f}')

        prod_str = ' + '.join([
            f'{signal[k-(n_ker-1)+i]:.0f}×{ker_flipped[i]:.1f}'
            for i in range(n_ker)
            if 0 <= k - (n_ker - 1) + i < n_sig
        ])
        ax_top.set_title(f'Kernel at k={k}:  output[{k}] = {prod_str} = {output_val:.3f}',
                         fontsize=10)
        ax_top.set_ylabel('Amplitude')
        ax_top.set_xlim(-1, n_res)
        ax_top.set_ylim(-0.5, 4)
        ax_top.legend(fontsize=8, loc='upper right')
        ax_top.grid(True, alpha=0.3)

        # Legend for green bars
        ax_top.bar([], [], color='#2ecc71', label='Products')
        ax_top.legend(fontsize=8, loc='upper right')

        # ── Bottom: result built up so far ──────────────────────────────────
        partial = np.full(n_res, np.nan)
        partial[:k + 1] = result[:k + 1]

        ax_bot.bar(range(n_res), result, color='#cccccc', alpha=0.4,
                   label='Final result (preview)')
        ax_bot.bar(range(k + 1), result[:k + 1], color='#e74c3c', alpha=0.8,
                   label='Built so far')
        ax_bot.bar(k, result[k], color='#c0392b', alpha=1.0,
                   label=f'Current output[{k}] = {result[k]:.3f}')
        ax_bot.set_title('Convolution Output Built Up Step by Step')
        ax_bot.set_xlabel('Sample index n')
        ax_bot.set_ylabel('Amplitude')
        ax_bot.set_xlim(-1, n_res)
        ax_bot.legend(fontsize=8, loc='upper right')
        ax_bot.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    # Slider
    ax_slider = plt.axes([0.15, 0.06, 0.7, 0.03])
    slider = Slider(ax_slider, 'Position  k', 0, n_res - 1,
                    valinit=0, valstep=1, color='#4a90d9')
    slider.on_changed(draw)
    draw(0)
    plt.show()


def plot_convolution_in_audio():
    """
    Connects convolution back to audio filtering:
    shows three different kernels and their effect on the same signal,
    side by side in both time and frequency domains.
    """
    sr       = 44100
    duration = 0.05
    t        = np.linspace(0, duration, int(sr * duration))

    # Signal: mix of three tones
    y = (np.sin(2 * np.pi * 300  * t) +
         np.sin(2 * np.pi * 1500 * t) +
         np.sin(2 * np.pi * 5000 * t))

    nyquist = sr / 2
    kernels = {
        'Low-pass\n(keeps bass)'    : firwin(101,  500  / nyquist),
        'Band-pass\n(keeps mids)'   : firwin(101, [800  / nyquist, 3000 / nyquist], pass_zero=False),
        'High-pass\n(keeps treble)' : firwin(101,  3000 / nyquist, pass_zero=False),
    }

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    fig, axes = plt.subplots(len(kernels) + 1, 2, figsize=(14, 12))
    fig.suptitle('Convolution in Audio — Three Filters on the Same Signal',
                 fontsize=13, fontweight='bold')

    # Original signal row
    freqs     = np.fft.rfftfreq(len(y), 1 / sr)
    fft_orig  = np.abs(np.fft.rfft(y))

    axes[0, 0].plot(t * 1000, y, color='#4a90d9', lw=0.8)
    axes[0, 0].set_title('Original Signal (300 + 1500 + 5000 Hz)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_xlabel('Time (ms)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs, fft_orig, color='#4a90d9')
    axes[0, 1].set_title('Original Spectrum')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_xlim(0, 8000)
    axes[0, 1].grid(True, alpha=0.3)

    # Filtered rows
    for row, (name, h) in enumerate(kernels.items(), start=1):
        y_filt     = np.convolve(y, h, mode='same')
        fft_filt   = np.abs(np.fft.rfft(y_filt))
        col        = colors[row - 1]

        axes[row, 0].plot(t * 1000, y,      color='#4a90d9', alpha=0.3, lw=0.8,
                          label='Original')
        axes[row, 0].plot(t * 1000, y_filt, color=col,       lw=0.8,
                          label='Filtered')
        axes[row, 0].set_title(f'{name} — Time Domain')
        axes[row, 0].set_ylabel('Amplitude')
        axes[row, 0].set_xlabel('Time (ms)')
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(freqs, fft_orig, color='#4a90d9', alpha=0.3, label='Original')
        axes[row, 1].plot(freqs, fft_filt, color=col,                  label='Filtered')
        axes[row, 1].set_title(f'{name} — Frequency Domain')
        axes[row, 1].set_ylabel('Amplitude')
        axes[row, 1].set_xlabel('Frequency (Hz)')
        axes[row, 1].set_xlim(0, 8000)
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Step-by-step static explanation
    plot_convolution_explanation()

    # 2. Interactive slider — slide the kernel yourself
    plot_convolution_interactive()

    # 3. Back to audio — three filters on a real signal
    plot_convolution_in_audio()