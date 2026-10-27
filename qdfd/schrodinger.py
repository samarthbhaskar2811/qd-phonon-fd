"""Finite-difference solver for the effective-mass Schrodinger equation.

Units: lengths in nm, energies in meV. The kinetic prefactor is
hbar^2 / (2 m0) = 38.0998 meV nm^2, divided by the relative effective mass.

The Hamiltonian H = -(hbar^2 / 2m*) Laplacian + V(r) is discretised on a
uniform Cartesian grid with the standard second-order central difference and
Dirichlet (psi = 0) boundaries, so the discretisation error in the energies
scales as O(h^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HBAR2_OVER_2M0 = 38.09982  # meV nm^2


@dataclass(frozen=True)
class Grid:
    """Uniform grid centred on the origin, one entry per dimension."""

    n: tuple[int, ...]  # interior points per axis
    length: tuple[float, ...]  # box side per axis (nm); psi vanishes at the walls

    @classmethod
    def cube(cls, n: int, length: float, dim: int = 3) -> "Grid":
        return cls((n,) * dim, (length,) * dim)

    @property
    def dim(self) -> int:
        return len(self.n)

    @property
    def spacing(self) -> tuple[float, ...]:
        # n interior points plus two boundary points split the box into n + 1 intervals
        return tuple(L / (n + 1) for n, L in zip(self.n, self.length))

    @property
    def axes(self) -> list[np.ndarray]:
        return [(np.arange(1, n + 1) * h) - L / 2 for n, h, L in zip(self.n, self.spacing, self.length)]

    @property
    def mesh(self) -> list[np.ndarray]:
        return np.meshgrid(*self.axes, indexing="ij")

    @property
    def dv(self) -> float:
        return float(np.prod(self.spacing))

    def radius(self) -> np.ndarray:
        return np.sqrt(sum(x**2 for x in self.mesh))


def laplacian(grid: Grid) -> sp.csr_matrix:
    """Sparse N-D Laplacian as a Kronecker sum of 1-D second differences."""
    ops = []
    for n, h in zip(grid.n, grid.spacing):
        main = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        ops.append(sp.diags([off, main, off], [-1, 0, 1], format="csr") / h**2)

    lap = sp.csr_matrix((int(np.prod(grid.n)),) * 2)
    for axis, d2 in enumerate(ops):
        term = sp.identity(1, format="csr")
        for j, n in enumerate(grid.n):
            term = sp.kron(term, d2 if j == axis else sp.identity(n, format="csr"), format="csr")
        lap = lap + term
    return lap.tocsr()


def hamiltonian(grid: Grid, potential: np.ndarray, mass: float) -> sp.csr_matrix:
    """H = -(hbar^2 / 2 m* m0) Laplacian + V, with V in meV on the grid."""
    if potential.shape != grid.n:
        raise ValueError(f"potential shape {potential.shape} does not match grid {grid.n}")
    kinetic = -(HBAR2_OVER_2M0 / mass) * laplacian(grid)
    return (kinetic + sp.diags(potential.ravel())).tocsr()


def solve(grid: Grid, potential: np.ndarray, mass: float, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Lowest k eigenstates.

    Returns energies (meV, ascending) and wavefunctions with shape (k, *grid.n),
    normalised so that sum(|psi|^2) * dV = 1.
    """
    h = hamiltonian(grid, potential, mass)
    # Shift-invert just below the potential minimum targets the bottom of the spectrum.
    sigma = float(potential.min()) - 1.0
    energies, vecs = spla.eigsh(h, k=k, sigma=sigma, which="LM")
    order = np.argsort(energies)
    energies, vecs = energies[order], vecs[:, order]

    psis = vecs.T.reshape((k, *grid.n))
    norms = np.sqrt((np.abs(psis) ** 2).sum(axis=tuple(range(1, grid.dim + 1))) * grid.dv)
    psis = psis / norms.reshape((k,) + (1,) * grid.dim)
    return energies, psis


# --- confinement potentials --------------------------------------------------

def spherical_well(grid: Grid, radius: float, depth: float) -> np.ndarray:
    """Finite spherical well: -depth inside r < radius, 0 outside (meV)."""
    return np.where(grid.radius() < radius, -depth, 0.0)


def harmonic(grid: Grid, mass: float, hbar_omega: float) -> np.ndarray:
    """Isotropic harmonic potential (1/2) m* omega^2 r^2 for a given hbar*omega (meV)."""
    # (1/2) m w^2 r^2 = (hbar w)^2 r^2 / (4 * hbar^2/(2m))
    return (hbar_omega**2) * grid.radius() ** 2 / (4.0 * HBAR2_OVER_2M0 / mass)


def harmonic_length(mass: float, hbar_omega: float) -> float:
    """Oscillator length l = sqrt(hbar / (m* omega)) in nm; |psi_0|^2 ~ exp(-r^2 / l^2)."""
    return float(np.sqrt(2.0 * HBAR2_OVER_2M0 / (mass * hbar_omega)))
