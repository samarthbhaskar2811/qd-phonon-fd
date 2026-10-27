"""Acoustic-phonon spectral density of a quantum-dot exciton.

Longitudinal-acoustic (LA) phonons couple to the electron and hole through the
deformation potential. For an exciton made of electron and hole densities
rho_e(r), rho_h(r), the spectral density is

    J(w) = w^3 / (4 pi^2 rho hbar c^5) * < |D_e F_e(q) - D_h F_h(q)|^2 >_angles,   q = w / c

with form factors F(q) = integral rho(r) exp(i q.r) d^3r, mass density rho,
LA sound velocity c and deformation potentials D_e, D_h. For identical
Gaussian densities |psi|^2 ~ exp(-r^2 / l^2) this reduces to the familiar

    J(w) = (D_e - D_h)^2 w^3 / (4 pi^2 rho hbar c^5) * exp(-w^2 l^2 / (2 c^2)).

Here J is returned in ps^-1 as a function of hbar*w in meV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schrodinger import Grid

HBAR = 1.054571817e-34  # J s
EV = 1.602176634e-19  # J
MEV_TO_RAD_PER_S = 1e-3 * EV / HBAR  # hbar*w [meV] -> w [rad/s]


@dataclass(frozen=True)
class Material:
    """Bulk parameters for LA deformation-potential coupling."""

    density: float  # kg / m^3
    sound_velocity: float  # LA, m / s
    d_e: float  # electron deformation potential, eV
    d_h: float  # hole deformation potential, eV


# GaAs values as commonly used in the QD dephasing literature
# (e.g. Krummheuer, Axt and Kuhn, PRB 65, 195313 (2002)). Check them against
# your source of choice before quoting results.
GAAS = Material(density=5370.0, sound_velocity=5110.0, d_e=-14.6, d_h=-4.8)


def fibonacci_sphere(n: int) -> np.ndarray:
    """n near-uniform unit vectors, used for the angular average over q."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], axis=1)


def form_factor(density: np.ndarray, grid: Grid, q_vectors: np.ndarray, chunk: int = 256) -> np.ndarray:
    """F(q) = sum_r rho(r) exp(i q.r) dV, by direct summation on the grid.

    density: |psi|^2 on the grid (normalised to 1), q_vectors: (M, 3) in nm^-1.
    """
    if grid.dim != 3:
        raise ValueError("form factors need a 3-D grid")
    x, y, z = (c.ravel() for c in grid.mesh)
    weights = density.ravel() * grid.dv
    out = np.empty(len(q_vectors), dtype=complex)
    for start in range(0, len(q_vectors), chunk):
        q = q_vectors[start : start + chunk]
        phase = np.outer(q[:, 0], x) + np.outer(q[:, 1], y) + np.outer(q[:, 2], z)
        out[start : start + chunk] = np.exp(1j * phase) @ weights
    return out


def _prefactor(material: Material, w: np.ndarray) -> np.ndarray:
    """w^3 / (4 pi^2 rho hbar c^5) in units of J^-2 s^-1, for w in rad/s."""
    return w**3 / (4.0 * np.pi**2 * material.density * HBAR * material.sound_velocity**5)


def spectral_density(
    rho_e: np.ndarray,
    rho_h: np.ndarray,
    grid: Grid,
    hbar_omega: np.ndarray,
    material: Material = GAAS,
    n_directions: int = 64,
) -> np.ndarray:
    """J(w) in ps^-1 for hbar*w in meV, from electron and hole densities on a 3-D grid."""
    w = np.asarray(hbar_omega, dtype=float) * MEV_TO_RAD_PER_S
    q_mag = w / material.sound_velocity * 1e-9  # m^-1 -> nm^-1
    dirs = fibonacci_sphere(n_directions)
    q_vectors = (q_mag[:, None, None] * dirs[None, :, :]).reshape(-1, 3)

    f_e = form_factor(rho_e, grid, q_vectors).reshape(len(w), n_directions)
    f_h = form_factor(rho_h, grid, q_vectors).reshape(len(w), n_directions)
    coupling = np.mean(np.abs(material.d_e * EV * f_e - material.d_h * EV * f_h) ** 2, axis=1)
    return _prefactor(material, w) * coupling * 1e-12  # s^-1 -> ps^-1


def gaussian_spectral_density(
    hbar_omega: np.ndarray, l_e: float, l_h: float, material: Material = GAAS
) -> np.ndarray:
    """Analytic J(w) (ps^-1) for Gaussian densities |psi|^2 ~ exp(-r^2 / l^2), lengths in nm."""
    w = np.asarray(hbar_omega, dtype=float) * MEV_TO_RAD_PER_S
    q = w / material.sound_velocity * 1e-9
    coupling = (material.d_e * np.exp(-(q * l_e) ** 2 / 4) - material.d_h * np.exp(-(q * l_h) ** 2 / 4)) ** 2 * EV**2
    return _prefactor(material, w) * coupling * 1e-12


def reorganisation_energy(hbar_omega: np.ndarray, j: np.ndarray) -> float:
    """Polaron shift  hbar * integral J(w)/w dw, in meV (J in ps^-1, hbar*w in meV)."""
    w = np.asarray(hbar_omega) * MEV_TO_RAD_PER_S
    integrand = np.divide(j * 1e12, w, out=np.zeros_like(w), where=w > 0)
    return float(HBAR * np.trapezoid(integrand, w) / (1e-3 * EV))
