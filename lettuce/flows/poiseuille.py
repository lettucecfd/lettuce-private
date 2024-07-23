"""
Poiseuille Flow
"""

import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary


class PoiseuilleFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice, initialize_with_zeros=True):
        self.resolution = resolution
        self.lattice = lattice
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.initialize_with_zeros = initialize_with_zeros

    def analytic_solution(self, grid):
        half_lattice_spacing = 0.5 / self.resolution
        x, y = grid
        nu = self.units.viscosity_pu
        rho = 1
        u1 = torch.tensor(self.acceleration[0] / (2 * rho * nu)
                          * ((y - half_lattice_spacing)
                             * (1 - half_lattice_spacing - y)),
                          device=x.device)
        u2 = torch.zeros_like(x)
        u = torch.stack((u1, u2), dim=0)
        p = torch.zeros_like(x) + self.units.convert_density_lu_to_pressure_pu(rho)
        return p, u

    def initial_solution(self, grid):
        if self.initialize_with_zeros:
            p = torch.zeros((1, *grid[0].shape), dtype=torch.float,
                            device=grid[0].device)
            u = torch.zeros((2, *grid[0].shape), dtype=torch.float,
                            device=grid[0].device)
            return p, u
        else:
            return self.analytic_solution(grid)

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution + 1, endpoint=True)
        y = np.linspace(0, 1, num=self.resolution + 1, endpoint=True)
        return self.units.lattice.convert_to_tensor(np.meshgrid(x, y,
                                                                indexing='ij'))

    @property
    def boundaries(self):
        mask = np.zeros(self.grid[0].shape, dtype=bool)
        mask[:, [0, -1]] = True
        boundary = BounceBackBoundary(mask=mask, lattice=self.units.lattice)
        return [boundary]

    @property
    def acceleration(self):
        return np.array([0.001, 0])
