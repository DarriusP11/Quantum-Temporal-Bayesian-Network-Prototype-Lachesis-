"""Path-integral (worldline) QMC demo for 1D transverse-field Ising model.

This is a classical Monte Carlo simulation of a quantum system using
Trotter decomposition. It is NOT a quantum circuit algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class QMCResults:
    magnetization_z: float
    energy_like: float
    acceptance_rate: float
    params: dict


def _trotter_couplings(beta: float, j: float, gamma: float, m_slices: int) -> Tuple[float, float]:
    """Return effective couplings for the anisotropic 2D Ising mapping.

    K_space = beta * J / M
    K_time  = 0.5 * log(coth(beta * Gamma / M))
    """
    k_space = beta * j / m_slices
    x = beta * gamma / m_slices
    # Avoid division by zero for very small x.
    x = max(x, 1e-12)
    k_time = 0.5 * math.log(math.cosh(x) / math.sinh(x))
    return k_space, k_time


def run_tfim_qmc(
    n_spins: int = 8,
    beta: float = 2.0,
    j: float = 1.0,
    gamma: float = 1.0,
    m_slices: int = 20,
    sweeps: int = 2000,
    thermalization: int = 500,
    seed: int | None = 0,
) -> QMCResults:
    """Worldline QMC with single-spin Metropolis updates.

    Returns a simple "energy-like" estimator from the 2D Ising action
    and the z-magnetization proxy.
    """
    if n_spins < 2 or m_slices < 2:
        raise ValueError("n_spins and m_slices must be >= 2")
    if thermalization >= sweeps:
        raise ValueError("thermalization must be less than sweeps")

    rng = np.random.default_rng(seed)
    k_space, k_time = _trotter_couplings(beta, j, gamma, m_slices)

    # Worldline spins: shape (n_spins, m_slices), values in {-1, +1}
    spins = rng.choice([-1, 1], size=(n_spins, m_slices))

    total_accepts = 0
    total_attempts = 0

    mag_sum = 0.0
    energy_like_sum = 0.0
    measurements = 0

    for sweep in range(sweeps):
        # Visit sites in random order each sweep
        for _ in range(n_spins * m_slices):
            i = rng.integers(0, n_spins)
            t = rng.integers(0, m_slices)

            s = spins[i, t]
            # Periodic boundary conditions
            s_left = spins[(i - 1) % n_spins, t]
            s_right = spins[(i + 1) % n_spins, t]
            s_prev = spins[i, (t - 1) % m_slices]
            s_next = spins[i, (t + 1) % m_slices]

            # Local action change for flipping s
            delta = 2.0 * s * (
                k_space * (s_left + s_right) + k_time * (s_prev + s_next)
            )

            total_attempts += 1
            if delta <= 0.0 or rng.random() < math.exp(-delta):
                spins[i, t] = -s
                total_accepts += 1

        if sweep >= thermalization:
            # Magnetization in z-basis proxy
            mag = spins.mean()
            # Energy-like estimator from the 2D Ising action
            space_term = 0.0
            time_term = 0.0
            for i in range(n_spins):
                for t in range(m_slices):
                    s = spins[i, t]
                    s_right = spins[(i + 1) % n_spins, t]
                    s_next = spins[i, (t + 1) % m_slices]
                    space_term += s * s_right
                    time_term += s * s_next

            e_like = -(k_space * space_term + k_time * time_term) / (n_spins * m_slices)

            mag_sum += mag
            energy_like_sum += e_like
            measurements += 1

    return QMCResults(
        magnetization_z=mag_sum / measurements,
        energy_like=energy_like_sum / measurements,
        acceptance_rate=total_accepts / max(total_attempts, 1),
        params={
            "n_spins": n_spins,
            "beta": beta,
            "j": j,
            "gamma": gamma,
            "m_slices": m_slices,
            "sweeps": sweeps,
            "thermalization": thermalization,
            "k_space": k_space,
            "k_time": k_time,
        },
    )


def main() -> None:
    results = run_tfim_qmc()
    print("Path-integral QMC (TFIM) demo")
    print(f"magnetization_z: {results.magnetization_z:.4f}")
    print(f"energy_like: {results.energy_like:.4f}")
    print(f"acceptance_rate: {results.acceptance_rate:.3f}")
    print("params:", results.params)


if __name__ == "__main__":
    main()
