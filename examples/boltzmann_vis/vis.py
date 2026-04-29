#!/usr/bin/env python3

"""Plot Boltzmann weighting-derived occupancy vs relative free energy."""

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # kcal/(mol*K), common thermochemistry value
    r = 0.0019872041
    t = 298.15

    # Relative free energy range in kcal/mol.
    dg = np.linspace(-5.0, 5.0, 500)
    w = np.exp(-dg / (r * t))
    y = w / (1.0 + w)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dg, y, color="tab:blue", linewidth=2)
    ax.set_xlabel("Relative free energy, ΔG (kcal/mol)")
    ax.set_ylabel("Boltzmann-weighted fraction, w / (1 + w)")
    ax.set_title("Boltzmann weighting as a function of ΔG")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
