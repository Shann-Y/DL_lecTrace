"""Plot helpers for the SF-Swin-UNETR lecture. The underscore keeps this file
out of the trace (same convention as the lectrace demo)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def signal_plot(t, x, title):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.plot(t, x, color="tab:blue", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("position")
    ax.set_ylabel("intensity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def spectrum_plot(freq, mag, title):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.stem(freq, mag, basefmt=" ")
    ax.set_title(title)
    ax.set_xlabel("frequency")
    ax.set_ylabel("magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
