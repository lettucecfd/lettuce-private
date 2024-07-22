import os.path
import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from examples.occ_geometries.geometric_building_model import build_house
from examples.occ_geometries.obstacleFunctions import (calculate_mask,
                                                       makeGrid,
                                                       collect_collision_data)
from examples.occ_geometries.obstacleSurface import ObstacleSurface
from examples.postprocessing.plotting import (collect_intersections,
                                              plot_intersections)
import lettuce as lt
from lettuce.boundary import CollisionData
import matplotlib.pyplot as plt
import matplotlib as mpl

# default_device = torch.device("cuda:0" if torch.cuda.is_available() else
# "cpu")
default_device = torch.device("cpu")
torch.set_default_device(default_device)
print(f"Doing calculations on {default_device}")

scaling = 20
pos_x, pos_y, height, width, rel_roof_dx, rel_roof_dy = (
    scaling*10, scaling*2, scaling*3, scaling*10, 0.1, 0.2)
parameters_primal = torch.tensor([pos_x, pos_y, height, width, rel_roof_dx,
                                  rel_roof_dy])  # solid point inside box
parameters_tangent = torch.zeros_like(parameters_primal)
parameters_tangent[0] = 1.
with fwAD.dual_level():
    parameters = fwAD.make_dual(parameters_primal, parameters_tangent)
    pos_x, pos_y, height, width, rel_roof_dx, rel_roof_dy = parameters
    occ_data = build_house(pos_x, pos_y, height, width, rel_roof_dx,
                           rel_roof_dy)
    shape = (50*scaling, 10*scaling)
    domain_constraints = ([0, 0], [shape[0], shape[1]])
    lattice = lt.Lattice(lt.D2Q9, device=default_device)
    grid = makeGrid(domain_constraints, shape)
    coll_data = CollisionData()
    coll_data = calculate_mask(occ_data, grid, collision_data=coll_data)
    coll_data = collect_collision_data(occ_data, coll_data, lattice, grid)
    d_gt = coll_data.d_gt
    print(f"output = ", d_gt)  # should be a 0D-tensor with 1 Element and
    # tangent=sth.
    if len(d_gt) == 0:
        print(f"output.item() = ", d_gt.item())
    else:
        print(f"output.tolist() = ", d_gt.tolist())
    print(f"fwAD.unpack_dual(output).tangent = ",
          fwAD.unpack_dual(d_gt).tangent)
    print("   If this is 'tensor(0.1234), the gradient was lost somewhere "
          "along the way but set at transform back to "
          "tensor. If it is None, it was not transformed back.")
    print(f"output.tolist() = ", d_gt.tolist())
    d_grad = np.array(fwAD.unpack_dual(coll_data.d_lt).tangent.tolist() +
                      fwAD.unpack_dual(coll_data.d_gt).tangent.tolist())
    flow = ObstacleSurface(res=scaling, shape=shape, reynolds_number=200,
                           mach_number=0.01, u_init=1, lattice=lattice,
                           char_length_pu=1, char_velocity_pu=1,
                           domain_constraints=domain_constraints, depth=1.)
    flow.add_boundary(coll_data, lt.InterpolatedBounceBackBoundary,
                      name='house_ad')
    collision = lt.BGKCollision(lattice,
                                tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision,
                               streaming=streaming)
    if not os.path.exists("TestForwardFlow"):
        os.mkdir("TestForwardFlow")
    simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=10,
                                               filename_base="TestForwardFlow/"
                                                             "vtk"))
    simulation.initialize_f_neq()
    u0max = flow.ux.max().item()
    simulation.step(100)
    u = torch.norm(flow.units.convert_velocity_to_pu(lattice.u(simulation.f)),
                   dim=0)
    n = 100
    for _ in range(n):
        simulation.step(1)
        u += torch.norm(flow.units.convert_velocity_to_pu(
                lattice.u(simulation.f)
            ), dim=0)
    u /= n+1
    u_grad = fwAD.unpack_dual(u).tangent

fluid_coords, dir_coords, surface_coords = collect_intersections(coll_data,
                                                                 grid, lattice)

fluid_x, fluid_y, fluid_z = fluid_coords
dir_x, dir_y, dir_z = dir_coords
surface_x, surface_y, surface_z = surface_coords

# quiver plot in pyplot
fig, ax = plt.subplots()
if d_grad is None or abs(d_grad.mean() - 0.1234) < 0.0001:
    d_grad = np.array(coll_data.d_lt.tolist() + coll_data.d_gt.tolist())
norm = mpl.colors.Normalize(vmin=d_grad.min(), vmax=d_grad.max())
cm = mpl.cm.coolwarm
sm = mpl.cm.ScalarMappable(cmap=cm, norm=norm)
sm.set_array([])
ax.quiver(fluid_x, fluid_y, dir_x, dir_y, color=cm(d_grad), angles='xy',
          scale_units='xy', scale=1, label='IBB vectors')
fig.colorbar(sm, ax=ax)
ax.scatter(grid[0][coll_data.solid_mask].cpu(),
           grid[1][coll_data.solid_mask].cpu(),
           color='k', s=.5, alpha=0.4, marker='.', label='solid_mask')
ax.scatter(grid[0].cpu(), grid[1].cpu(), s=.5, alpha=0.2, marker='.',
           label='grid')
if u_grad is not None:
    ax.imshow(lattice.convert_to_numpy(u_grad).transpose())
else:
    ax.imshow(lattice.convert_to_numpy(u).transpose())
    print("u_grad is None!")
ax.scatter(surface_x, surface_y, s=1, marker='.', label='intersection points')
ax.axis('equal')
plt.show()
