"""
Boundary Conditions.

The `__call__` function of a boundary defines its application to the distribution functions.

Boundary conditions can define a mask (a boolean numpy array)
that specifies the grid points on which the boundary
condition operates.

Boundary classes can define two functions `make_no_streaming_mask` and `make_no_collision_mask`
that prevent streaming and collisions on the boundary nodes.

The no-stream mask has the same dimensions as the distribution functions (Q, x, y, (z)) .
The no-collision mask has the same dimensions as the grid (x, y, (z)).

"""

# To Do:
#  - the inits for Halfway and Fullway Bounce Back with force calculation (neighbor search) can be outsourced to a function taking mask and lattice and returning tensor(f_mask)
#  - same for the calc_force_on_boundary method
#  - fullway and halfway bounce back could be fitted into one class and specified by parameter (hw/fw) determining the use of how call acts and if no_stream is used (hw)

import torch
import numpy as np
import time

# from build.lib.lettuce.boundary import InterpolatedBounceBackBoundary
from lettuce import LettuceException
from lettuce.lattices import Lattice

__all__ = ["LettuceBoundary", "BounceBackBoundary", "AntiBounceBackOutlet",
           "EquilibriumBoundaryPU", "EquilibriumOutletP", "SlipBoundary",
           "InterpolatedBounceBackBoundary", "PartiallySaturatedBoundary",]


class LettuceBoundary:
    mask: torch.Tensor
    no_streaming_mask: torch.Tensor
    lattice: Lattice


class PartiallySaturatedBoundary(LettuceBoundary):
    """
    Partially saturated boundary condition using a partial combination of standard full-way bounce back and
    BGK-Collision, first presented by Noble and Torczynski (1998), see Krüger et al., pp. 448.
    """

    # this may be just as efficient as a compact version, b/c the boundary is actually used on all nodes even within the object
    def __init__(self, mask: torch.Tensor, lattice: Lattice, tau: float, saturation: float):
        self.mask = mask
        self.lattice = lattice
        self.tau = tau
        self.B = saturation * (tau - 0.5) / ((1 - saturation) + (tau - 0.5))  # B(epsilon, theta), Krüger p. 448ff
        return

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f, rho=rho)
        feq = self.lattice.equilibrium(rho, u)
        # TODO: benchmark and possibly use indices (like _compact)
        #  and/or calculate feq twice within torch.where (like _less_memory)
        f = torch.where(self.mask, f - (1.0 - self.B) / self.tau * (f - feq)
                        + self.B * ((f[self.lattice.stencil.opposite] - feq[self.lattice.stencil.opposite])
                                    - (f - self.lattice.equilibrium(rho, torch.zeros_like(u)))), f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class CollisionData(dict):
    f_index_lt: torch.IntTensor
    f_index_gt: torch.IntTensor
    d_lt: torch.Tensor
    d_gt: torch.Tensor
    points_inside: torch.Tensor
    solid_mask: torch.Tensor
    not_intersected: torch.Tensor = torch.tensor([])


class InterpolatedBounceBackBoundary(LettuceBoundary):
    """Interpolated Bounce Back Boundary Condition first introduced by Bouzidi et al. (2001), as described in Kruger et al.
        (2017)
        - linear or quadratic interpolation of populations to retain the true boundary location between fluid- and
        solid-node
        * version 2.0: using given indices and distances between fluid- and solid-node
        of boundary link and boundary surface for interpolation!
    """

    def __init__(self, mask, lattice: Lattice, collision_data: CollisionData, calc_force=None, ad_enabled=False):
        t_init_start = time.time()
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.ad_enabled = ad_enabled
        if calc_force is not None:
            self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
                self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
            self.calc_force = True
        else:
            self.calc_force = False

        # convert relevant tensors:
        self.f_index_lt = torch.tensor(collision_data.f_index_lt, device=self.lattice.device,
                                       dtype=torch.int)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(collision_data.f_index_gt, device=self.lattice.device,
                                       dtype=torch.int)  # the batch-index has to be integer
        self.d_lt = collision_data.d_lt
        self.d_gt = collision_data.d_gt
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int)  # batch-index has to be a tensor

        f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)
        print(f"IBB initialization took {time.time() - t_init_start:.2f} seconds")

    def __call__(self, f):
        ## f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.lattice.D == 2:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[
                    self.f_index_lt[:, 0],
                    self.f_index_lt[:, 1],
                    self.f_index_lt[:, 2]]
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        if self.lattice.D == 3:
            # BOUNCE
            # if d <= 0.5
            if len(self.f_index_lt) != 0:
                f[self.opposite_tensor[self.f_index_lt[:, 0]],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2],
                self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[
                    self.f_index_lt[:, 0],
                    self.f_index_lt[:, 1],
                    self.f_index_lt[:, 2],
                    self.f_index_lt[:, 3]]
            # if d > 0.5
            if len(self.f_index_gt) != 0:
                f[self.opposite_tensor[self.f_index_gt[:, 0]],
                self.f_index_gt[:, 1],
                self.f_index_gt[:, 2],
                self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        # CALC. FORCE on boundary (MEM, MEA)
        if self.calc_force:
            self.calc_force_on_boundary(f)
        return f

    def make_no_streaming_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_streaming_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        # return torch.tensor(self.mask, dtype=torch.bool)
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        # return torch.tensor(self.mask, dtype=torch.bool)  # self.lattice.convert_to_tensor(self.mask)
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opp.])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2]],
                                          self.lattice.e[self.f_index_lt[:, 0]].float()) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2]],
                                            self.lattice.e[self.f_index_gt[:, 0]].float())
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2],
                                              self.f_index_lt[:, 3]],
                                          self.lattice.e[self.f_index_lt[:, 0]].float()) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2],
                                                self.f_index_gt[:, 3]],
                                            self.lattice.e[self.f_index_gt[:, 0]].float())

    def store_f_collided(self, f_collided):
        for f_index_lgt, f_collided_lgt in zip([self.f_index_lt, self.f_index_gt],
                                               [self.f_collided_lt, self.f_collided_gt]):
            if len(f_index_lgt) != 0:
                for d in range(self.lattice.D):
                    indices = [f_index_lgt[:, 0],  # q
                               f_index_lgt[:, 1],  # x
                               f_index_lgt[:, 2]]  # y
                    if self.lattice.D == 3:
                        indices.append(f_index_lgt[:, 3])
                    f_collided_lgt[:, 0] = torch.clone(f_collided[indices])
                    indices[0] = self.opposite_tensor[f_index_lgt[:, 0]]
                    f_collided_lgt[:, 1] = torch.clone(f_collided[indices])


class SlipBoundary(LettuceBoundary):
    """bounces back in a direction given as 0, 1, or 2 for x, y, or z, respectively
        based on fullway bounce back algorithm (population remains in the wall for 1 time step)
    """

    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.bb_direction = direction
        e = self.lattice.stencil.e
        bb_direction = self.bb_direction
        opposite_stencil = np.array(e)
        opposite_stencil[:, bb_direction] = -e[:, bb_direction]
        self.opposite = []
        for opp_dir in opposite_stencil:
            self.opposite.append(np.where(np.array(e == opp_dir).all(axis=1))[0][0])

    def __call__(self, f):
        f = torch.where(self.mask, f[self.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class BounceBackBoundary(LettuceBoundary):
    """Fullway Bounce-Back Boundary"""

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class EquilibriumBoundaryPU(LettuceBoundary):
    # TODO: Make EQ BC compact
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """

    def __init__(self, mask, lattice, units, velocity, pressure=0):
        # parameter input (u, p) in PU!
        # u can be a field (individual ux, uy, (uz) for all boundary nodes) or vector (uniform ux, uy, (uz)))
        self.mask = mask  # lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)
        self.pressure = lattice.convert_to_tensor(pressure)

    def __call__(self, f):
        # convert PU-inputs to LU, calc feq and overwrite f with feq where mask==True
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask, feq, f)
        return f


class AntiBounceBackOutlet(LettuceBoundary):
    """Allows distributions to leave domain unobstructed through this boundary.
        Based on equations from page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, direction):

        assert isinstance(direction, list), \
            LettuceException(
                f"Invalid direction parameter. Expected direction of type list but got {type(direction)}.")

        assert len(direction) in [1, 2, 3], \
            LettuceException(
                f"Invalid direction parameter. Expected direction of of length 1, 2 or 3 but got {len(direction)}.")

        assert (direction.count(0) == (len(direction) - 1)) and ((1 in direction) ^ (-1 in direction)), \
            LettuceException(
                "Invalid direction parameter. "
                f"Expected direction with all entries 0 except one 1 or -1 but got {direction}.")

        direction = np.array(direction)
        self.lattice = lattice

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(
            np.argwhere(np.matmul(lattice.convert_to_numpy(self.lattice.stencil.e), direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        # construct indices for einsum and get w in proper shape for the calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
                - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
                (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
                 - (torch.norm(u_w, dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_streaming_mask(self, f_shape):
        no_streaming_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_streaming_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_streaming_mask

    # not 100% sure about this. But collisions seem to stabilize the boundary.
    # def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """

    def __init__(self, lattice, direction, rho0=1.0):
        super(EquilibriumOutletP, self).__init__(lattice, direction)
        self.rho0 = rho0

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = self.rho0 * torch.ones_like(rho[here])
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[..., None], u_w[..., None])[..., 0]
        return f

    def make_no_streaming_mask(self, f_shape):
        no_streaming_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_streaming_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_streaming_mask

    def make_no_collision_mask(self, f_shape):
        no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask
