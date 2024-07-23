"""
Couette Flow
"""

import numpy as np
import torch

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU


class CouetteFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        # TODO
        raise NotImplementedError

    def initial_solution(self, x):
        return (torch.zeros((1, self.resolution, self.resolution),
                            dtype=torch.float, device=x.device),
                torch.zeros((2, self.resolution, self.resolution),
                            dtype=torch.float, device=x.device))

    @property
    def grid(self):
        x = np.linspace(0, 1, num=self.resolution, endpoint=False)
        y = np.linspace(0, 1, num=self.resolution, endpoint=False)
        return self.units.lattice.convert_to_tensor(np.meshgrid(x, y,
                                                                indexing='ij'))

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(torch.abs(y - 1) < 1e-6,
                                      self.units.lattice, self.units,
                                      np.array([1.0, 0.0])),
                BounceBackBoundary(torch.abs(y) < 1e-6, self.units.lattice)]
