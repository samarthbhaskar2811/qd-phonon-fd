"""Validate the phonon spectral density against the analytic Gaussian result."""

import numpy as np
import pytest

from qdfd import GAAS, Grid, gaussian_spectral_density, harmonic, harmonic_length, reorganisation_energy, solve, spectral_density
from qdfd.phonons import EV, HBAR, MEV_TO_RAD_PER_S, fibonacci_sphere, form_factor

OMEGA = np.linspace(0.01, 4.0, 60)  # hbar*w in meV


def gaussian_density(grid, length):
    return np.exp(-grid.radius() ** 2 / length**2) / (np.pi * length**2) ** 1.5


def test_form_factor_is_one_at_zero_q():
    grid = Grid.cube(25, 30.0)
    rho = gaussian_density(grid, 4.0)
    f0 = form_factor(rho, grid, np.zeros((1, 3)))
    assert f0[0] == pytest.approx(1.0, abs=1e-6)


def test_form_factor_of_gaussian_is_gaussian():
    grid, length = Grid.cube(31, 36.0), 4.0
    rho = gaussian_density(grid, length)
    q = np.linspace(0.0, 1.0, 12)[:, None] * fibonacci_sphere(3)[0]
    q_mag = np.linalg.norm(q, axis=1)
    np.testing.assert_allclose(form_factor(rho, grid, q).real, np.exp(-(q_mag**2) * length**2 / 4), atol=1e-6)


def test_spectral_density_matches_analytic_for_gaussian_densities():
    grid = Grid.cube(31, 36.0)
    l_e, l_h = 4.0, 3.0
    numeric = spectral_density(gaussian_density(grid, l_e), gaussian_density(grid, l_h), grid, OMEGA)
    exact = gaussian_spectral_density(OMEGA, l_e, l_h)
    np.testing.assert_allclose(numeric, exact, rtol=1e-4, atol=1e-9 * exact.max())


def test_spectral_density_from_solved_harmonic_ground_states():
    # Electron and hole in harmonic traps: the FD ground states are Gaussians of known width.
    # Trap energies chosen so both widths are ~6 nm and the box holds > 4 widths each side.
    grid = Grid.cube(33, 50.0)
    me, mh, hw_e, hw_h = 0.067, 0.5, 30.0, 4.0
    _, psi_e = solve(grid, harmonic(grid, me, hw_e), me)
    _, psi_h = solve(grid, harmonic(grid, mh, hw_h), mh)
    numeric = spectral_density(psi_e[0] ** 2, psi_h[0] ** 2, grid, OMEGA)
    exact = gaussian_spectral_density(OMEGA, harmonic_length(me, hw_e), harmonic_length(mh, hw_h))
    # Residual error comes from the O(h^2) discretisation of the wavefunctions.
    np.testing.assert_allclose(numeric, exact, rtol=0.03, atol=0.01 * exact.max())


def test_reorganisation_energy_matches_closed_form():
    # Equal widths: J = A w^3 exp(-w^2 / wc^2), so hbar * int J/w dw = hbar A sqrt(pi) wc^3 / 4.
    length = 4.0
    omega = np.linspace(0.0, 15.0, 4000)
    j = gaussian_spectral_density(omega, length, length)
    a = (GAAS.d_e - GAAS.d_h) ** 2 * EV**2 / (4 * np.pi**2 * GAAS.density * HBAR * GAAS.sound_velocity**5)
    wc = np.sqrt(2) * GAAS.sound_velocity / (length * 1e-9)
    exact_mev = HBAR * a * np.sqrt(np.pi) * wc**3 / 4 / (1e-3 * EV)
    assert reorganisation_energy(omega, j) == pytest.approx(exact_mev, rel=1e-4)
    assert MEV_TO_RAD_PER_S > 0
