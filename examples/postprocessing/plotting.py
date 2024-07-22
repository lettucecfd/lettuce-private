import os.path
from math import floor
from time import time

import torch
from matplotlib import pyplot as plt, rcParams
from matplotlib.colors import colorConverter, LinearSegmentedColormap
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from lettuce import Lattice, Simulation
from lettuce.boundary import CollisionData

rcParams["image.aspect"] = 'equal'
rcParams["image.interpolation"] = 'none'
rcParams["image.origin"] = 'lower'


class Show2D:
    def __init__(self, lattice: Lattice, mask: torch.Tensor, domain_constraints, outdir: str, p_mask=None,
                 dpi: int = 600, save: bool = True, show: bool = True, show_mask: bool = True, mask_alpha=1,
                 figsize: tuple = (20, 4), transparent: bool = False, subplots=False):
        self.lattice = lattice
        self.outdir = outdir
        if len(mask.shape) > 2:
            mask = mask[:, :, int(mask.shape[2] / 2)]
        self.mask = self.lattice.convert_to_numpy(mask).transpose()
        minx, maxx = [_ for _ in domain_constraints]
        self.extent = (minx[0], maxx[0], minx[1], maxx[1])
        self.dpi = dpi
        self.save = save
        self.transparent = transparent
        self.show = show
        self.show_mask = show_mask
        self.subplots = subplots
        if p_mask is not None:
            if len(p_mask.shape) > 2:
                p_mask = p_mask[:, :, int(p_mask.shape[2] / 2)]
            self.p_mask = self.lattice.convert_to_numpy(p_mask).transpose()
        self.mask_alpha = mask_alpha
        self.figsize = figsize
        self.__call__(mask, "solid_mask", "solid_mask")

    def __call__(self, data, title: str, name: str, vlim: tuple[float, float] = None):
        fig, ax = plt.subplots(2, figsize=self.figsize) if self.subplots else plt.subplots(figsize=self.figsize)
        if len(data.shape) > 2:
            data = data[:, :, int(data.shape[2] / 2)]
        vmin, vmax = vlim if vlim is not None else None, None
        ax0 = ax[0] if self.subplots else ax
        p = ax0.imshow(self.lattice.convert_to_numpy(data).transpose(), extent=self.extent, vmin=vmin, vmax=vmax)
        if self.subplots:
            ax[1].imshow(self.lattice.convert_to_numpy(data).transpose(), extent=self.extent, vmin=vmin, vmax=vmax)
        if self.show_mask:
            b = colorConverter.to_rgba('white')
            w = colorConverter.to_rgba('black')
            cmap_solid = LinearSegmentedColormap.from_list('my_cmap', [b, w], 256)
            cmap_solid._init()  # create the _lut array, with rgba values
            cmap_solid._lut[:, -1] = np.linspace(0, 1, cmap_solid.N + 3)
            ax0.imshow(self.mask, extent=self.extent, cmap=cmap_solid, vmin=0, vmax=1)
            cmap_partial = LinearSegmentedColormap.from_list('my_cmap1', [b, w], 256)
            cmap_partial._init()  # create the _lut array, with rgba values
            cmap_partial._lut[:, -1] = np.linspace(0, self.mask_alpha, cmap_solid.N + 3)
            if hasattr(self, 'p_mask'):
                ax0.imshow(self.p_mask, extent=self.extent, cmap=cmap_partial, vmin=0, vmax=1)
        fig.suptitle(title)
        fig.colorbar(p, ax=ax)
        if self.show:
            plt.show()
        if self.save:
            fig.savefig(os.path.join(self.outdir, name), dpi=self.dpi, transparent=self.transparent)
        plt.close()


def print_results(lattice: Lattice, simulation: Simulation, res: float, dim: int, flow, steps: int, time0, mlups_avg,
                  show2d: Show2D, u0max, do_rho=True, do_vort=True, do_u=True):
    energy = torch.sum(lattice.incompressible_energy(simulation.f)) / (res ** dim)
    t = flow.units.convert_time_to_pu(steps)
    u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f))
    umaxLU = torch.norm(lattice.u(simulation.f), dim=0).max() / lattice.cs.item()
    time1 = time() - time0
    print(
        f"Step: {steps}, Time: {t:.1f} s, Energy: {energy:.2f}, MLUPS: {mlups_avg:.1f}, maxMa: {umaxLU:.4f}, runtime: "
        f"{floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
    if do_u:
        show2d(torch.norm(u, dim=0), f"u(it={steps},t={t:.1f}) [m/s]", f"u_{steps}", vlim=(-.2, u0max))
    if do_rho:
        rho = flow.units.convert_density_lu_to_pressure_pu(lattice.rho(simulation.f))  # [Pa]
        show2d(torch.norm(rho, dim=0) * 10 ** -5, f"density(it={steps},t={t:.1f}) [bar]", f"rho_{steps}",
               vlim=(-.002, .002))
    if do_vort:
        grad_u0 = torch.gradient(u[0])
        grad_u1 = torch.gradient(u[1])
        vorticity = (grad_u1[0] - grad_u0[1])
        show2d(torch.abs(vorticity), f"vorticity(it={steps},t={t:.1f})", f"vort_{steps}", vlim=(-.2, 1))


def collect_intersections(collision_data: CollisionData, grid: tuple[torch.Tensor, ...], lattice: Lattice):
    dim = len(grid[0].shape)
    # get the interpolated surface points
    surface_x, surface_y, surface_z = [], [], []
    fluid_x, fluid_y, fluid_z = [], [], []
    dir_x, dir_y, dir_z = [], [], []
    xstep = (grid[0][1, 0] - grid[0][0, 0]).item() \
        if dim == 2 else (grid[0][1, 0, 0] - grid[0][0, 0, 0]).item()
    ystep = (grid[1][0, 1] - grid[1][0, 0]).item() \
        if dim == 2 else (grid[1][0, 1, 0] - grid[1][0, 0, 0]).item()
    zstep = (grid[2][0, 0, 1] - grid[2][0, 0, 0]).item() if dim == 3 else None
    gridstep = (xstep, ystep) if dim == 2 else (xstep, ystep, zstep)

    f_index = collision_data.f_index_lt.tolist() + collision_data.f_index_gt.tolist()
    d = collision_data.d_lt.tolist() + collision_data.d_gt.tolist()

    e = lattice.e.tolist()
    for i in range(len(f_index)):
        iq, ix, iy, iz = f_index[i]
        if iz != 0:  # now, f_index is 3D
            pass
        dx, dy = [d[i] * e[iq][_] * gridstep[_] for _ in range(2)]
        f_x = grid[0][ix, iy].item() if dim == 2 else grid[0][ix, iy, iz].item()
        f_y = grid[1][ix, iy].item() if dim == 2 else grid[1][ix, iy, iz].item()
        dir_x.append(dx)
        dir_y.append(dy)
        fluid_x.append(f_x)
        fluid_y.append(f_y)
        surface_x.append(f_x + dx)
        surface_y.append(f_y + dy)
        if dim == 3:
            dz = d[i] * e[iq][2] * gridstep[2]
            f_z = grid[2][ix, iy, iz].item()
            dir_z.append(dz)
            fluid_z.append(f_z)
            surface_z.append(f_z + dz)
    fluid_coords = (fluid_x, fluid_y, fluid_z)
    dir_coords = (dir_x, dir_y, dir_z)
    surface_coords = (surface_x, surface_y, surface_z)
    return fluid_coords, dir_coords, surface_coords


def plot_not_intersected(collision_data: CollisionData, grid: tuple[torch.Tensor, ...], outdir: str, name: str):
    if not hasattr(collision_data, 'not_intersected'):
        print('Collision data has no not_intersected field!')
        return
    dim = len(grid[0].shape)
    # get the interpolated surface points
    xstep = (grid[0][1, 0] - grid[0][0, 0]) if dim == 2 else (grid[0][1, 0, 0] - grid[0][0, 0, 0])
    ystep = (grid[1][0, 1] - grid[1][0, 0]) if dim == 2 else (grid[1][0, 1, 0] - grid[1][0, 0, 0])
    zstep = (grid[2][0, 0, 1] - grid[2][0, 0, 0]) if dim == 3 else None
    # gridstep = (xstep, ystep) if dim == 2 else (xstep, ystep, zstep)

    fluid_x = collision_data.not_intersected[:, 0]
    fluid_y = collision_data.not_intersected[:, 1]
    # fluid_z = collision_data.not_intersected[:, 2] if dim == 3 else None
    dir_x = collision_data.not_intersected[:, 3] * xstep
    dir_y = collision_data.not_intersected[:, 4] * ystep
    # dir_z = collision_data.not_intersected[:, 5] * zstep if dim == 3 else None

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.scatter(grid[0].cpu(), grid[1].cpu(), s=1, alpha=0.4, color='c', label='grid')  # show whole grid
    ax.scatter(grid[0][collision_data.solid_mask].cpu(), grid[1][collision_data.solid_mask].cpu(),
               color='k', s=4, alpha=0.4, label='solid_mask')  # show solid_mask
    ax.quiver(fluid_x.cpu().numpy(), fluid_y.cpu().numpy(), dir_x.cpu().numpy(), dir_y.cpu().numpy(),
              angles='xy', scale_units='xy', scale=1, color='orange',
              # , marker=".", s=.5, color='orange',
              label='not intersected vectors')  # show intersection points
    ax.axis('equal')
    ax.legend()
    fig.savefig(os.path.join(outdir, f"{name}_not_intersected_{'3d' if dim == 3 else ''}.png"), dpi=600)
    plt.show()
    return


# plot the grid, interpolated surface points, and original surface
def plot_intersections(grid, mask, fluid_coords, dir_coords, surface_coords: list, outdir, name: str,
                       quiver=False, dim=2, show_grid=False, show_mask=True, show=True):
    fluid_x, fluid_y, fluid_z = fluid_coords
    dir_x, dir_y, dir_z = dir_coords
    if dim == 2:
        fig, ax = plt.subplots(figsize=(20, 4))
        if show_grid:
            ax.scatter(grid[0].cpu(), grid[1].cpu(), s=1, alpha=0.4, color='c', label='grid')  # show whole grid
        if show_mask:
            ax.scatter(grid[0][mask].cpu(), grid[1][mask].cpu(),
                       color='k', s=4, alpha=0.4, label='landscape_mask')  # show solid_mask
        if quiver:
            ax.quiver(fluid_x, fluid_y, dir_x, dir_y, angles='xy', scale_units='xy', scale=1, color='orange',
                      label='intersection vectors')  # show intersection points
        else:
            for surface_x, surface_y, surface_z in surface_coords:
                ax.scatter(surface_x, surface_y, marker=".", s=.5,
                           label='intersection points')  # show intersection points
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if show_grid:
            ax.scatter(grid[0].cpu(), grid[2].cpu(), grid[1].cpu(), s=1, alpha=0.4, color='c',
                       label='grid')  # show whole grid
        if show_mask:
            ax.scatter(grid[0][mask].cpu(), grid[2][mask].cpu(), grid[1][mask].cpu(),
                       color='k', s=4, alpha=0.4, label='solid_mask')  # show solid_mask
        for surface_x, surface_y, surface_z in surface_coords:
            ax.scatter(surface_x, surface_z, surface_y, marker=".", s=.5,
                       label='intersection points')  # show intersection points
        ax.set(xlim=(grid[0].min().item(), grid[0].max().item()),
               ylim=(grid[1].min().item(), grid[1].max().item()),
               zlim=(grid[2].min().item(), grid[2].max().item()))
    ax.axis('equal')
    ax.legend()
    fig.savefig(os.path.join(outdir,
                             f"{name}_{'quiver' if quiver else 'dots'}{'3d' if dim == 3 else ''}"
                             f"{'grid' if show_grid else ''}.png"), dpi=600)
    if show:
        plt.show()
    plt.close('all')


def plot_intersection_info(collisions: CollisionData or list[CollisionData], grid, lattice, full_mask, outdir,
                           name: str = '', show=True):
    if len(name) > 0:
        name += '_'
    ad_collision = collisions[0] if len(collisions) > 0 else collisions
    fluid_coords, dir_coords, surface_coords = collect_intersections(ad_collision, grid, lattice)

    if len(grid[0].shape) == 2:
        plot_intersections(grid, full_mask, fluid_coords, dir_coords, [surface_coords], outdir,
                           name=name + 'mask_2D_quiver', quiver=True, show=show)
        plot_intersections(grid, full_mask, fluid_coords, dir_coords, [surface_coords], outdir,
                           name=name + 'mask_2D_no_grid', show=show)
    else:
        plot_intersections(grid, full_mask, fluid_coords, dir_coords, [surface_coords], outdir,
                           name=name + 'mask_3D_no_grid', dim=3, show=show)
        plot_intersections(grid, full_mask, fluid_coords, dir_coords, [surface_coords], outdir,
                           name=name + 'mask_3D_and_grid', dim=3, show_grid=True, show=show)

    # count occurences of q's (stencil directions) in indices
    occurences_gt = torch.bincount(ad_collision.f_index_gt[:, 0]).tolist()
    occurences_lt = torch.bincount(ad_collision.f_index_lt[:, 0]).tolist()
    for iq in range(len(lattice.e)):
        occurences = occurences_gt[iq] if iq < len(occurences_gt) else 0
        occurences += occurences_lt[iq] if iq < len(occurences_lt) else 0
        print(f"Direction {[f'{_:+}' for _ in lattice.e[iq]]} occurs {occurences} times in ad collision info.")

    if len(collisions) > 1:
        perm_collision = collisions[1]
        fluid_coords2, dir_coords2, surface_coords2 = collect_intersections(perm_collision, grid, lattice)
        plot_intersections(grid, full_mask, fluid_coords, dir_coords, [surface_coords, surface_coords2],
                           outdir, name=name + 'mask_visualized_overlapped', show=show)
        # count occurences of q's (stencil directions) in indices
        occurences_gt = torch.bincount(perm_collision.f_index_gt[:, 0]).tolist()
        occurences_lt = torch.bincount(perm_collision.f_index_lt[:, 0]).tolist()
        for iq in range(len(lattice.e)):
            occurences = occurences_gt[iq] if iq < len(occurences_gt) else 0
            occurences += occurences_lt[iq] if iq < len(occurences_lt) else 0
            print(f"Direction {[f'{_:+}' for _ in lattice.e[iq]]} occurs {occurences} times in permanent collision "
                  f"info.")

    return


def plot_flow_field(surface_coords, lattice, field_data, xmax, ymax, label, name, image_folder, cluster,
                    do_colorbar: bool = True, title: str = None, transparent: bool = True):
    surface_x, surface_y, surface_z = surface_coords
    fig, ax = plt.subplots(figsize=(14, 3))
    # scatter-plots of solid mask, grid, and intersection points
    ax.scatter(surface_x, surface_y, s=2, marker='.', label='intersection points')
    # imshow of velocity-tangent or velocity
    u_plot = ax.imshow(lattice.convert_to_numpy(field_data).transpose(), interpolation='none',
                       extent=(0, xmax, 0, ymax))

    ### colorbar
    if do_colorbar:
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.7)
        bar_u = fig.colorbar(u_plot, cax=cax2)
        bar_u.set_label(label)
    if title is not None:
        ax.set_title(title)
    # nice axes and show
    ax.axis('equal')
    fig.savefig(os.path.join(image_folder, name), dpi=600, transparent=transparent)
    if not cluster:
        plt.show()
    plt.close("all")
