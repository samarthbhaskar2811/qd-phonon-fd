"""Reproduce every figure and number quoted in the README.

Run from the repository root:  python examples/run_all.py
Figures go to figures/, numbers to figures/results.txt.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdfd import Grid, gaussian_spectral_density, harmonic, harmonic_length, reorganisation_energy, solve, spectral_density, spherical_well
from qdfd.schrodinger import HBAR2_OVER_2M0
from tests.test_schrodinger import spherical_well_ground_state

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)
lines: list[str] = []

ME, MH = 0.067, 0.5  # GaAs electron, isotropic heavy-hole approximation
OMEGA = np.linspace(0.01, 5.0, 120)  # hbar*w, meV


def log(text: str) -> None:
    print(text)
    lines.append(text)


# 1. Convergence: 2-D harmonic oscillator (exact E0 = hw) and 3-D spherical well
hw = 20.0
h2, err2 = [], []
for n in (31, 45, 63, 89, 127):
    grid = Grid.cube(n, 90.0, dim=2)
    e, _ = solve(grid, harmonic(grid, ME, hw), ME)
    h2.append(grid.spacing[0])
    err2.append(abs(e[0] - hw) / hw)

radius, depth = 5.0, 200.0
exact_sw = spherical_well_ground_state(radius, depth, ME)
h3, err3 = [], []
for n in (19, 25, 31, 37):
    grid = Grid.cube(n, 22.0)
    e, _ = solve(grid, spherical_well(grid, radius, depth), ME)
    h3.append(grid.spacing[0])
    err3.append(abs(e[0] - exact_sw) / abs(exact_sw))

order2 = np.polyfit(np.log(h2), np.log(err2), 1)[0]
log(f"2-D harmonic oscillator: fitted convergence order {order2:.2f} (expected 2)")
log(f"3-D spherical well (R={radius} nm, V0={depth} meV): exact E0 = {exact_sw:.2f} meV, "
    f"FD relative error {err3[-1] * 100:.2f}% at h = {h3[-1]:.2f} nm")

fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(h2, err2, "o-", label="2-D harmonic oscillator")
ax.loglog(h3, err3, "s-", label="3-D spherical well")
ax.loglog(h2, err2[0] * (np.array(h2) / h2[0]) ** 2, "k--", lw=0.8, label=r"$\propto h^2$")
ax.set_xlabel("grid spacing h (nm)")
ax.set_ylabel("relative error in ground-state energy")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "convergence.png", dpi=150)

# 2. Validation of J(w): FD harmonic ground states vs the analytic Gaussian formula
grid = Grid.cube(33, 50.0)
_, psi_e = solve(grid, harmonic(grid, ME, 30.0), ME)
_, psi_h = solve(grid, harmonic(grid, MH, 4.0), MH)
j_fd = spectral_density(psi_e[0] ** 2, psi_h[0] ** 2, grid, OMEGA)
j_exact = gaussian_spectral_density(OMEGA, harmonic_length(ME, 30.0), harmonic_length(MH, 4.0))
peak_err = abs(j_fd.max() - j_exact.max()) / j_exact.max()
log(f"J(w) validation (harmonic traps): peak J differs from analytic by {peak_err * 100:.2f}%")

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(OMEGA, j_exact, "k-", label="analytic (Gaussian)")
ax.plot(OMEGA[::4], j_fd[::4], "o", ms=4, label="finite difference")
ax.set_xlabel(r"$\hbar\omega$ (meV)")
ax.set_ylabel(r"$J(\omega)$ (ps$^{-1}$)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "spectral_density_validation.png", dpi=150)

# 3. Size dependence for spherical-well dots (illustrative confinement depths)
fig, (ax_j, ax_psi) = plt.subplots(1, 2, figsize=(10, 4))
grid = Grid.cube(31, 22.0)
for r in (3.0, 4.0, 5.0):
    e_e, psi_e = solve(grid, spherical_well(grid, r, 300.0), ME)
    e_h, psi_h = solve(grid, spherical_well(grid, r, 150.0), MH)
    j = spectral_density(psi_e[0] ** 2, psi_h[0] ** 2, grid, OMEGA)
    lam = reorganisation_energy(OMEGA, j)
    w_peak = OMEGA[np.argmax(j)]
    log(f"R = {r:.0f} nm: E_e = {e_e[0]:.1f} meV, E_h = {e_h[0]:.1f} meV, "
        f"J peaks at {w_peak:.2f} meV, polaron shift (to 5 meV) = {lam * 1e3:.1f} ueV")
    ax_j.plot(OMEGA, j, label=f"R = {r:.0f} nm")
    mid = grid.n[2] // 2
    ax_psi.plot(grid.axes[0], psi_e[0][:, mid, mid] ** 2, label=f"electron, R = {r:.0f} nm")

ax_j.set_xlabel(r"$\hbar\omega$ (meV)")
ax_j.set_ylabel(r"$J(\omega)$ (ps$^{-1}$)")
ax_j.set_title("LA-phonon spectral density")
ax_j.legend()
ax_psi.set_xlabel("x (nm)")
ax_psi.set_ylabel(r"$|\psi_e(x,0,0)|^2$ (nm$^{-3}$)")
ax_psi.set_title("Electron ground-state density")
ax_psi.legend()
fig.tight_layout()
fig.savefig(OUT / "size_dependence.png", dpi=150)

(OUT / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
