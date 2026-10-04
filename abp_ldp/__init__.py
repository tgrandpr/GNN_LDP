"""Static large deviations of the subvolume density in 2D active Brownian particles.

The large parameter is the subvolume area v = ell^2 (not time):
    P(N_v = rho v) ~ exp[-v I(rho)],   psi(lambda) = lim (1/v) ln E[exp(lambda N_v)].
"""

from .simulator import ABPParams, Geometry, ABPPopulation, count_in_square, count_in_squares

__all__ = ["ABPParams", "Geometry", "ABPPopulation", "count_in_square", "count_in_squares"]
