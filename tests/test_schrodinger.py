"""Validate the finite-difference solver against problems with exact answers."""

import numpy as np
import pytest
from scipy.optimize import brentq

from qdfd import Grid, harmonic, solve, spherical_well
from qdfd.schrodinger import HBAR2_OVER_2M0

GAAS_ME = 0.067  # GaAs electron effective mass


def test_infinite_well_1d_matches_analytic():
    length, mass = 10.0, GAAS_ME
    grid = Grid.cube(400, length, dim=1)
    energies, _ = solve(grid, np.zeros(grid.n), mass, k=4)
    n = np.arange(1, 5)
    exact = n**2 * np.pi**2 * (HBAR2_OVER_2M0 / mass) / length**2
    np.testing.assert_allclose(energies, exact, rtol=1e-3)


def test_harmonic_1d_ladder():
    mass, hw = GAAS_ME, 20.0
    grid = Grid.cube(600, 120.0, dim=1)
    energies, _ = solve(grid, harmonic(grid, mass, hw), mass, k=5)
    np.testing.assert_allclose(energies, (np.arange(5) + 0.5) * hw, rtol=1e-3)


def test_wavefunctions_are_normalised_and_orthogonal():
    mass, hw = GAAS_ME, 20.0
    grid = Grid.cube(80, 80.0, dim=2)
    _, psis = solve(grid, harmonic(grid, mass, hw), mass, k=3)
    overlap = np.einsum("aij,bij->ab", psis, psis) * grid.dv
    np.testing.assert_allclose(overlap, np.eye(3), atol=1e-8)


def test_harmonic_2d_degeneracy():
    # 2-D isotropic oscillator: E = (n + 1) hw with degeneracy n + 1.
    mass, hw = GAAS_ME, 20.0
    grid = Grid.cube(90, 90.0, dim=2)
    energies, _ = solve(grid, harmonic(grid, mass, hw), mass, k=6)
    np.testing.assert_allclose(energies, [1, 2, 2, 3, 3, 3] * np.array(hw), rtol=5e-3)


def test_harmonic_2d_converges_at_second_order():
    # Box half-width is 6 oscillator lengths, so truncation error is negligible
    # and only the O(h^2) discretisation error remains.
    mass, hw, length = GAAS_ME, 20.0, 90.0
    errors, spacings = [], []
    for n in (45, 63, 89):
        grid = Grid.cube(n, length, dim=2)
        energy, _ = solve(grid, harmonic(grid, mass, hw), mass, k=1)
        errors.append(abs(energy[0] - 1.0 * hw))
        spacings.append(grid.spacing[0])
    order = np.polyfit(np.log(spacings), np.log(errors), 1)[0]
    assert order == pytest.approx(2.0, abs=0.15)


def spherical_well_ground_state(radius, depth, mass):
    """Exact s-state energy of a finite spherical well: k cot(kR) = -kappa."""
    t = HBAR2_OVER_2M0 / mass

    def mismatch(e):
        k, kappa = np.sqrt((e + depth) / t), np.sqrt(-e / t)
        return k * np.cos(k * radius) + kappa * np.sin(k * radius)

    # The ground state has kR in (pi/2, pi), i.e. E between these limits.
    lower = max(-depth + 1e-9, (np.pi / (2 * radius)) ** 2 * t - depth)
    upper = min(-1e-9, (np.pi / radius) ** 2 * t - depth)
    return brentq(mismatch, lower, upper)


def test_spherical_well_3d_matches_analytic():
    radius, depth, mass = 5.0, 200.0, GAAS_ME
    exact = spherical_well_ground_state(radius, depth, mass)
    grid = Grid.cube(31, 22.0)
    energy, _ = solve(grid, spherical_well(grid, radius, depth), mass, k=1)
    # A sphere on a cubic grid has a staircase boundary, so agreement is ~1%, not 0.01%.
    assert energy[0] == pytest.approx(exact, rel=0.02)
