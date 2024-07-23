"""
Taylor-Green vortex in 2D and 3D.
"""

import numpy as np
import torch

from lettuce.unit import UnitConversion


class TaylorGreenVortex2D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution,
            characteristic_length_pu=2 * torch.pi,
            characteristic_velocity_pu=1
        )

    def analytic_solution(self, x, t=0):
        nu = torch.tensor(self.units.viscosity_pu)
        u = torch.stack((torch.cos(x[0]) * torch.sin(x[1]) * torch.exp(-2 * nu * t),
                      -torch.sin(x[0]) * torch.cos(x[1]) * torch.exp(-2 * nu * t)))
        p = -torch.tensor(0.25 * (torch.cos(2 * x[0]) + torch.cos(2 * x[1])) * torch.exp(-4 * nu * t))[None, :, :]
        return p, u

    def initial_solution(self, x):
        return self.analytic_solution(x, t=0)

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return self.units.lattice.convert_to_tensor(np.meshgrid(x, y,
                                                                indexing='ij'))

    @property
    def boundaries(self):
        return []


class TaylorGreenVortex3D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution / (2 * torch.pi), characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):
        u = torch.stack((
            torch.sin(x[0]) * torch.cos(x[1]) * torch.cos(x[2]),
            -torch.cos(x[0]) * torch.sin(x[1]) * torch.cos(x[2]),
            torch.zeros_like(x[0])
        ))
        p = (1 / 16. * (torch.cos(2 * x[0]) + torch.cos(2 * x[1])) * (torch.cos(2 * x[2]) + 2))[None,:,:,:]
        return p, u

    @property
    def grid(self):
        x = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        z = np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        return self.units.lattice.convert_to_tensor(np.meshgrid(x, y, z,
                                                                indexing='ij'))

    @property
    def boundaries(self):
        return []
