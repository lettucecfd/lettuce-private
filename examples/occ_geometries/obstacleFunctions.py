import multiprocessing
import os
import time
from math import floor, sqrt

import numpy as np
import torch
import trimesh
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Precision import precision
from OCC.Core.TopAbs import TopAbs_ON, TopAbs_IN
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Compound
from joblib import Parallel, delayed

from examples.differentiable_geometry_REQUIRES_DLR_CODE import occ_ad
from examples.occ_geometries.geometry import intersect_boundary_with_ray, extract_faces
from examples.postprocessing.plotting import plot_not_intersected
from examples.differentiable_geometry_REQUIRES_DLR_CODE.occ_ad import standard_adouble_list_to_torch_tensor, OCC_has_ad, make_gp_Pnt
from lettuce.boundary import CollisionData
from lettuce.lattices import Lattice
if OCC_has_ad:
    from OCC.Core.ForwardAD import Standard_Adouble

def calculate_mask(boundary_object: TopoDS_Solid or TopoDS_Shape or trimesh.Trimesh, grid: tuple[torch.Tensor, ...],
                   name: str = 'no_name', collision_data: CollisionData = None, cut_z: float = 0) -> CollisionData:
    is_occ = type(boundary_object) is TopoDS_Shape or type(boundary_object) is TopoDS_Solid
    is_tri = type(boundary_object) is trimesh.Trimesh
    print(f"calculate_points_inside for '{name}'.")
    time0 = time.time()
    ndim = len(grid)
    z = cut_z if ndim == 2 else None
    nx, ny, *nz = grid[0].shape
    nz = nz[0] if nz else 1
    n_all = nx * ny * nz
    if is_occ:
        bounding_box = Bnd_Box()
        brepbndlib.Add(boundary_object, bounding_box)
        solidClassifier = BRepClass3d_SolidClassifier(boundary_object)
        tol = Standard_Adouble(precision.Confusion()) if OCC_has_ad else precision.Confusion()
        points_inside = []
        # looping through all indices (if D==2, iz is only 0 and z remains 0.)
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if ndim == 2:
                        x, y = [grid[_][ix, iy].item() for _ in [0, 1]]
                    else:
                        x, y, z = [grid[_][ix, iy, iz].item() for _ in [0, 1, 2]]
                    tmp_pnt = make_gp_Pnt(x, y, z)
                    if not bounding_box.IsOut(tmp_pnt):
                        solidClassifier.Perform(tmp_pnt, tol)
                        state = solidClassifier.State()
                        if state == TopAbs_ON or state == TopAbs_IN:
                            points_inside.append([ix, iy, iz])
                    # progress update
                    count += 1
                    if count % min(int(n_all / 5) + 1, 50000) == 0:
                        print(f"calculate_mask {count}/{n_all} ({count / n_all:.2%}), "
                              f"ix={ix}, iy={iy}, iz={iz}, {len(points_inside)} points inside so far.")
        solid_mask = mask_from_points_list(points_inside, grid, ndim, name)
        points_inside = torch.tensor(points_inside)
    elif is_tri and boundary_object.is_watertight and name not in ['terrain', 'surface'] and 'landscape' not in name:
        # contains() yields also bottom points so landscape will be done with rays
        print(f"Boundary object '{name}' is watertight. Using Trimesh.contains().")
        points_list = (np.vstack([grid[0].cpu().numpy().ravel(),
                                  grid[1].cpu().numpy().ravel(),
                                  grid[2].cpu().numpy().ravel() if ndim == 3 else grid[1].cpu().numpy().ravel() * 0]
                                 ).T.reshape(-1, 3))
        id_grid = np.meshgrid(np.arange(ny), np.arange(nx), np.arange(nz))
        points_id_list = np.vstack([id_grid[1].ravel(),
                                    id_grid[0].ravel(),
                                    id_grid[2].ravel()]).T.reshape(-1, 3)
        time1 = time.time()
        points_are_inside = boundary_object.contains(points_list[:, [0, 2, 1]])
        print(f"Trimesh.contains() took {time.time() - time1:.1f} secs for {len(points_list)} grid points. "
              f"Found {sum(points_are_inside)} solid points.")
        points_inside = torch.tensor(points_id_list[points_are_inside])
        solid_mask = torch.tensor(points_are_inside.reshape(grid[0].shape))
    else:
        print(f"Boundary object '{name}' is not watertight or a surface mesh. Intersecting rays from top.")
        points_inside = []
        # looping through all indices (if D==2, iz is only 0 and z remains 0.)
        ymax = grid[1].max().item()
        count = 0
        for ix in range(nx):
            xi = grid[0][ix, 0, 0].item() if ndim == 3 else grid[0][ix, 0].item()
            for iz in range(nz) if ndim == 3 else [None]:
                zi = grid[2][0, 0, iz].item() if ndim == 3 else cut_z
                top_point = [[xi, zi, ymax]]  # swapped y and z for trimesh to understand
                # create ray in upwards-direction for each point
                downward_ray = [[0., 0., -1.]]
                # run the mesh-ray query
                intersection, _, _ = boundary_object.ray.intersects_location(ray_origins=top_point,
                                                                             ray_directions=downward_ray,
                                                                             multiple_hits=False)
                if len(intersection) > 0:
                    y_intersect = intersection[:, 2].max()
                    # solid_mask[ix,:,iz] = torch.where(grid[1][ix, :, iz] < y_intersect, 1, 0)
                    for iy in range(ny):
                        yi = grid[1][0, iy, 0].item() if ndim == 3 else grid[1][0, iy].item()
                        if yi < y_intersect:
                            points_inside.append([ix, iy, iz if ndim == 3 else 0])
                        # progress update
                        count += 1
                        if count % min(int(n_all / 5) + 1, 50000) == 0:
                            print(f"is_point_inside_solid {count}/{n_all} ({count / n_all:.2%}), "
                                  f"ix={ix}, iy={iy}, iz={iz}, {len(points_inside)} points inside so far.")
        solid_mask = mask_from_points_list(points_inside, grid, ndim, name)
        points_inside = torch.tensor(points_inside)
    n_inside = torch.sum(solid_mask).item()
    n_outside = torch.sum(~solid_mask).item()
    print(f"calculate_mask finished. Found {n_inside} points inside.")
    time1 = time.time() - time0
    print(f"Search for points inside '{name}' took "
          f"{floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
    if n_inside == 0:
        print(f"WARNING: No points inside '{name}' found. This should not happen.")
    if n_inside + n_outside == 0:
        print(f"WARNING: No intersection points found for '{name}'. This should not happen.")
    if n_inside + n_outside < solid_mask.shape[0] * solid_mask.shape[1] * (
            solid_mask.shape[2] if len(solid_mask.shape) > 2 else 1):
        print(f"WARNING: No intersection points found for '{name}'. This should not happen.")
    assert solid_mask.shape == grid[0].shape
    collision_data = CollisionData() if collision_data is None else collision_data
    collision_data.solid_mask, collision_data.points_inside = solid_mask, points_inside
    return collision_data


def mask_from_points_list(points: list[list[int]], grid: tuple[torch.Tensor, ...], ndim: int,
                          name: str = 'no_name') -> torch.Tensor:
    time0 = time.time()
    # create solid_mask
    solid_mask = torch.zeros_like(grid[0], dtype=torch.bool)
    if ndim == 2:  # expanding to 3D
        solid_mask = solid_mask[:, :, None]
    for point_inside_indices in points:
        ix, iy, iz = point_inside_indices
        solid_mask[ix, iy, iz] = True
    if ndim == 2:  # reducing back to 2D
        solid_mask = solid_mask[:, :, 0]
    time1 = time.time() - time0
    print(f"Conversion to solid_mask for '{name}' took "
          f"{floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
    return solid_mask


def neighbour_search(i_point_inside: int, points_inside: torch.Tensor, this_stencil: torch.Tensor, time0: float,
                     n_points_inside: int, solid_mask: torch.Tensor, opposite: list, ndim: int,
                     grid: tuple[torch.Tensor, ...], name: str = 'no_name', is_occ: bool = False,
                     is_tri: bool = False, shell: TopoDS_Compound = None, cut_z: float = 0, cluster: bool = False):
    point_index = points_inside[i_point_inside]
    time1 = time.time() - time0
    if i_point_inside % (min(int(n_points_inside / 5) + 1, 5000) + 1) == 0:
        print(f"neighbour_search {i_point_inside}/{n_points_inside} "
              f"({i_point_inside / n_points_inside:.2%}, "
              f"{floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss])")
    f_index_tmp, d_tmp = [], []
    fluid_point_tmp, fluid_point_id_tmp, ray_tmp, iq_fluid_tmp = [], [], [], []
    for neighbour_dir in this_stencil:
        neighbour_point_index = point_index + neighbour_dir  # jump to neighbor node
        # check if the neighbor node is in fluid
        # disregarding rays across domain borders
        border_crossed = False
        for dim in range(ndim):
            if not 0 <= neighbour_point_index[dim] < solid_mask.shape[dim]:
                border_crossed = True
        if not border_crossed:
            ix, iy, iz = neighbour_point_index[0], neighbour_point_index[1], neighbour_point_index[
                2] if ndim == 3 else None
            is_fluid = ~solid_mask[ix, iy, iz]
        else:
            is_fluid = False
        if is_fluid:
            # iq = stencil index; get the stencil index from the point of view of the solid node
            iq_solid = torch.where(torch.all(this_stencil == neighbour_dir, dim=1))[0].item()
            # add the direction in which the fluid node meets a solid node (first entry in f_index_d/lt
            iq_fluid = opposite[iq_solid]
            # fp = fluid fluid_point; sp = solid fluid_point
            # get the solid and fluid indices and coordinates
            ix_fp, iy_fp, iz_fp = neighbour_point_index.tolist()
            ix_sp, iy_sp, iz_sp = point_index.tolist()
            if ndim == 3:
                x_fp, y_fp, z_fp = [grid[dim][ix_fp, iy_fp, iz_fp].item() for dim in range(3)]
                x_sp, y_sp, z_sp = [grid[dim][ix_sp, iy_sp, iz_sp].item() for dim in range(3)]
            else:
                x_fp, y_fp = [grid[dim][ix_fp, iy_fp].item() for dim in range(2)]
                x_sp, y_sp = [grid[dim][ix_sp, iy_sp].item() for dim in range(2)]
                z_fp, z_sp = cut_z, cut_z
            if is_occ:
                fluid_point = make_gp_Pnt(x_fp, y_fp, z_fp)
                solid_point = make_gp_Pnt(x_sp, y_sp, z_sp)
                # get the intersection point and distance
                intersection_point, d = intersect_boundary_with_ray(fluid_point, solid_point, shell, cluster=cluster)
                if d < 0 or d > 1:
                    print(
                        f"WARNING: Ray between fluid point [{ix_fp},{iy_fp},{iz_fp}] "
                        f"at [{x_fp:.2f},{y_fp:.2f},{z_fp:.2f}] and solid point [{ix_sp},{iy_sp},{iz_sp}] "
                        f"at [{x_sp:.2f},{y_sp:.2f},{z_sp:.2f}] "
                        f"in direction {this_stencil[iq_fluid].tolist()} intersects boundary '{name}' only at "
                        f"{[round(_, 2) for _ in intersection_point.Coord()] if is_occ else [round(_, 2) for _ in intersection_point[0][[0, 2, 1]].tolist()]}. So, d={d:.2f}, "
                        f"but must be in [0, 1]. Setting d to 1.")
                    d = 1.
                f_index_tmp.append([iq_fluid, ix_fp, iy_fp, iz_fp])
                d_tmp.append(d)
            elif is_tri:
                fluid_point_tmp.append([x_fp, y_fp, z_fp])
                fluid_point_id_tmp.append([ix_fp, iy_fp, iz_fp])
                ray_tmp.append(this_stencil[iq_fluid].tolist())
                iq_fluid_tmp.append(iq_fluid)
    return [f_index_tmp, d_tmp, fluid_point_tmp, fluid_point_id_tmp, ray_tmp, iq_fluid_tmp]


def collect_collision_data(boundary_object: TopoDS_Solid or TopoDS_Shape or trimesh.Trimesh,
                           collision_data: CollisionData, lattice: Lattice, grid: tuple[torch.Tensor, ...],
                           name: str = 'no_name', parallel=False, outdir: str = os.getcwd(), cut_z: float = 0,
                           cluster: bool = False) -> CollisionData:
    is_occ = type(boundary_object) is TopoDS_Shape or type(boundary_object) is TopoDS_Solid
    is_tri = type(boundary_object) is trimesh.Trimesh
    # get a shell object to properly calculate minimum distances
    shell = extract_faces(boundary_object) if is_occ else None
    print(f"collect_collision_data for '{name}'.")
    ### search all neighbors of solid nodes for fluid nodes to be bounced back
    time0 = time.time()
    this_stencil = lattice.stencil.e.to(torch.tensor([0]).device)
    ndim = lattice.D
    if ndim == 2:  # adding a 0 for 3rd dimension
        this_stencil = torch.cat((this_stencil, torch.unsqueeze(torch.zeros_like(this_stencil[:, 0]), 1)), 1)
    f_index_lt, f_index_gt, d_lt, d_gt = [], [], [], []
    ds_stencil = []
    for direction in this_stencil:
        if ndim == 3:
            x1, y1, z1 = grid[0][1, 1, 1], grid[1][1, 1, 1], grid[2][1, 1, 1]
            x2, y2, z2 = (grid[0][1 + direction[0], 1 + direction[1], 1 + direction[2]],
                          grid[1][1 + direction[0], 1 + direction[1], 1 + direction[2]],
                          grid[2][1 + direction[0], 1 + direction[1], 1 + direction[2]])
            d12 = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            ds_stencil.append(d12)
        else:
            x1, y1 = grid[0][1, 1], grid[1][1, 1]
            x2, y2 = (grid[0][1 + direction[0], 1 + direction[1]],
                      grid[1][1 + direction[0], 1 + direction[1]])
            d12 = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            ds_stencil.append(d12)
    fluid_points, rays, iq_fluids, fluid_point_ids, solid_point_ids, solid_points = [], [], [], [], [], []
    n_points_inside = len(collision_data.points_inside)

    if parallel:
        print("Doing parallel run for neighbour search.")
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose=10)(
            delayed(neighbour_search)(i_direction, collision_data.points_inside, this_stencil, time0,
                                      n_points_inside, collision_data.solid_mask, lattice.stencil.opposite, ndim,
                                      grid, name, is_occ, is_tri, shell, cut_z, cluster)
            for i_direction in range(n_points_inside))
    else:
        print("Doing sequential run for neighbour search.")
        results = []
        for i_point_inside in range(n_points_inside):  # search all directions from
            result_tmp = neighbour_search(i_point_inside, collision_data.points_inside, this_stencil, time0,
                                          n_points_inside, collision_data.solid_mask, lattice.stencil.opposite, ndim,
                                          grid, name, is_occ, is_tri, shell, cut_z, cluster)
            if result_tmp is not None:
                results.append(result_tmp)

    for result in results:
        if is_occ:
            f_index_list, d_list = result[0], result[1]
            if len(d_list) > 0:
                for i in range(len(d_list)):
                    f_index, d = f_index_list[i], d_list[i]
                    if d <= 0.5:
                        f_index_lt.append(f_index)
                        d_lt.append(d)
                    else:
                        f_index_gt.append(f_index)
                        d_gt.append(d)
        elif is_tri:
            fluid_point_list, fluid_point_id_list, ray_list, iq_fluid_list = result[2], result[3], result[4], result[5]
            if len(ray_list) > 0:
                for i in range(len(ray_list)):
                    fluid_points.append(fluid_point_list[i])
                    fluid_point_ids.append(fluid_point_id_list[i])
                    rays.append(ray_list[i])
                    iq_fluids.append(iq_fluid_list[i])
    print(f"Collected {str(len(rays)) + ' rays.' if is_tri else str(len(d_lt) + len(d_gt)) + ' ds' if is_occ else ''}")
    if is_tri:
        fluid_points = np.array(fluid_points)
        rays = np.array(rays)
        time2 = time.time()
        print("Starting Trimesh.ray.intersects_location")
        n, step = 0, 100
        while n < len(rays):
            intersection_points_i, intersection_indices_i, _ = boundary_object.ray.intersects_location(
                ray_origins=fluid_points[n:n + step, [0, 2, 1]],
                ray_directions=rays[n:n + step, [0, 2, 1]],
                multiple_hits=False)
            if n == 0:
                intersection_points, intersection_indices = intersection_points_i, intersection_indices_i
            elif len(intersection_indices_i) > 0:
                intersection_points = np.append(intersection_points, intersection_points_i, axis=0)
                intersection_indices = np.append(intersection_indices, intersection_indices_i + n, axis=0)
            time1 = time.time() - time2
            n += step
            if n % min(int(len(rays) / 10) + 1, 5000) == 0:
                print(f"Intersected {min(n + step, len(rays))} ({min(n + step, len(rays)) / len(rays):.2%}, "
                      f"{floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss]) rays. "
                      f"Found {len(intersection_points)} intersections.")
        time1 = time.time() - time2
        print(f"Trimesh.ray.intersects_location took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss]")
        unique_indices = np.unique(intersection_indices)
        not_intersected = []
        for i in range(len(rays)):
            if i not in unique_indices:
                not_intersected_ray = fluid_points[i].tolist() + rays[i].tolist()
                not_intersected.append(not_intersected_ray)
        if not_intersected:
            print(f"WARNING: {len(not_intersected)} rays were not intersected.")
            collision_data.not_intersected = torch.tensor(not_intersected)
            plot_not_intersected(collision_data, grid, outdir, name)
        else:
            collision_data.not_intersected = torch.tensor([])
        for i in unique_indices:
            i_intersections = np.where(intersection_indices == i)[0]
            if len(i_intersections) > 0:
                ds = []
                for i_intersected in i_intersections:
                    intersection_index = intersection_indices[i_intersected]
                    x_ip, y_ip, z_ip = intersection_points[i_intersected][[0, 2, 1]]
                    x_fp, y_fp, z_fp = fluid_points[intersection_index]
                    ix_fp, iy_fp, iz_fp = fluid_point_ids[intersection_index]
                    iq_fluid = iq_fluids[intersection_index]
                    ds.append(
                        sqrt((x_fp - x_ip) ** 2 + (y_fp - y_ip) ** 2 + (z_fp - z_ip) ** 2) / ds_stencil[iq_fluid])
                d = min(ds)
                if 0 <= d <= 0.5:
                    f_index_lt.append([iq_fluid, ix_fp, iy_fp, iz_fp])
                    d_lt.append(d)
                elif d <= 1:
                    f_index_gt.append([iq_fluid, ix_fp, iy_fp, iz_fp])
                    d_gt.append(d)
                else:
                    f_index_gt.append([iq_fluid, ix_fp, iy_fp, iz_fp])
                    print(f"Ray {[iq_fluid, ix_fp, iy_fp, iz_fp]} at {[round(_, 2) for _ in [x_fp, y_fp, z_fp]]} "
                          f"has wrong length, with d={d}. Setting it to .999")
                    d_gt.append(.999)
            else:
                print(f"WARNING: No intersections found for intersection index {i}.")
        print(f"{len(d_lt)} ds in [0.,0.5], {len(d_gt)} ds in [0.5,1.)")
    # fixing fluid nodes in dents with d<=0.5 in both directions
    if len(f_index_lt) > 0:
        have_solid_opposite = []
        for i in range(len(f_index_lt)):
            iq, ix, iy, iz = f_index_lt[i]
            iqo = lattice.stencil.opposite[iq]
            if [iqo, ix, iy, iz] in f_index_lt or [iqo, ix, iy, iz] in f_index_gt:
                have_solid_opposite.append([iq, ix, iy, iz])
        for index_list in have_solid_opposite:
            iq, ix, iy, iz = index_list
            x, y, *z = [grid[_][ix, iy, iz if ndim == 3 else None].item() for _ in range(ndim)]
            z = z[0] if z else cut_z
            i_to_remove = [i for i in range(len(f_index_lt)) if f_index_lt[i] == index_list]
            d_tmp = [d_lt[_] for _ in i_to_remove]
            new_d = 0.50001
            print(f"Changing ray {index_list} at fluid node {[round(_, 2) for _ in [x, y, z]]} "
                  f"with d={d_tmp} from lt to gt with d={new_d}")
            del d_lt[i_to_remove[0]]
            del f_index_lt[i_to_remove[0]]
            d_gt.append(new_d)
            f_index_gt.append(index_list)
        print(f"{len(have_solid_opposite)} points contain f_index_lt in a direction and its opposite. "
              f"Removed lt entries and added gt entries to avoid interpolation with no_stream_points.")
    # store results in tensors
    f_index_lt = torch.tensor(f_index_lt, dtype=torch.int)
    f_index_gt = torch.tensor(f_index_gt, dtype=torch.int)
    d_lt = standard_adouble_list_to_torch_tensor(d_lt) if occ_ad.OCC_has_ad else torch.tensor(d_lt)
    d_gt = standard_adouble_list_to_torch_tensor(d_gt) if occ_ad.OCC_has_ad else torch.tensor(d_gt)
    time1 = time.time() - time0
    print(f"Neighbor search in '{name}' took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
    collision_data.f_index_lt = f_index_lt
    collision_data.f_index_gt = f_index_gt
    collision_data.d_lt = d_lt
    collision_data.d_gt = d_gt
    return collision_data


def makeGrid(domain_constraints, shape):
    dim = len(shape)
    xyz = tuple(torch.linspace(domain_constraints[0][_], domain_constraints[1][_], shape[_]) for _ in
                range(dim))  # tuple of lists of x,y,(z)-values/indices
    grid = torch.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (und z-)values/indices
    return grid


def overlap_solids(ad_collision_data: CollisionData, perm_collision_data: CollisionData):
    print("Doing overlap_ibb_solids")
    full_mask = perm_collision_data.solid_mask | ad_collision_data.solid_mask
    remove_ad_gt, remove_ad_lt, remove_perm_gt, remove_perm_lt = [], [], [], []
    time0 = time.time()
    ndim = len(full_mask.shape)
    for i_ad_gt in range(len(ad_collision_data.f_index_gt)):
        ad_index = ad_collision_data.f_index_gt[i_ad_gt]
        if full_mask[ad_index[1], ad_index[2], ad_index[3] if ndim == 3 else None]:
            remove_ad_gt.append(ad_index)
        elif torch.any(torch.all(perm_collision_data.f_index_lt == ad_index, dim=1)):
            remove_ad_gt.append(ad_index)
        else:
            in_perm_gt = torch.where(torch.all(perm_collision_data.f_index_gt == ad_index, dim=1))[0]
            if len(in_perm_gt) > 0:
                if ad_collision_data.d_gt[i_ad_gt] < perm_collision_data.d_gt[in_perm_gt]:
                    remove_perm_gt.append(in_perm_gt)
                else:
                    remove_ad_gt.append(ad_index)

    for i_ad_lt in range(len(ad_collision_data.f_index_lt)):
        ad_index = ad_collision_data.f_index_lt[i_ad_lt]
        if full_mask[ad_index[1], ad_index[2], ad_index[3] if ndim == 3 else None]:
            remove_ad_lt.append(ad_index)
        if torch.any(torch.all(perm_collision_data.f_index_gt == ad_index, dim=1)):
            remove_ad_lt.append(ad_index)
        else:
            in_perm_lt = torch.where(torch.all(perm_collision_data.f_index_lt == ad_index, dim=1))[0]
            if len(in_perm_lt) > 0:
                if ad_collision_data.d_lt[i_ad_lt] < perm_collision_data.d_lt[in_perm_lt]:
                    remove_perm_lt.append(in_perm_lt)
                else:
                    remove_ad_lt.append(ad_index)

    for i_perm_lt in range(len(perm_collision_data.f_index_lt)):
        perm_index = perm_collision_data.f_index_lt[i_perm_lt]
        if full_mask[perm_index[1], perm_index[2], perm_index[3] if ndim == 3 else None]:
            remove_perm_lt.append(perm_index)
    for i_perm_gt in range(len(perm_collision_data.f_index_gt)):
        perm_index = perm_collision_data.f_index_gt[i_perm_gt]
        if full_mask[perm_index[1], perm_index[2], perm_index[3] if ndim == 3 else None]:
            remove_perm_gt.append(perm_index)

    print(
        f"Removing {len(remove_ad_lt) + len(remove_ad_gt)}/{len(ad_collision_data.d_lt) + len(ad_collision_data.d_gt)} points from ad "
        f"and {len(remove_perm_lt) + len(remove_perm_gt)}/{len(perm_collision_data.d_lt) + len(perm_collision_data.d_gt)} points from permanent.")

    for remove_fs in remove_ad_gt:
        ad_collision_data.d_gt = ad_collision_data.d_gt[
            torch.where(~torch.all(ad_collision_data.f_index_gt == remove_fs, dim=1))]
        ad_collision_data.f_index_gt = ad_collision_data.f_index_gt[
            torch.where(~torch.all(ad_collision_data.f_index_gt == remove_fs, dim=1))]
    for remove_fs in remove_ad_lt:
        ad_collision_data.d_lt = ad_collision_data.d_lt[
            torch.where(~torch.all(ad_collision_data.f_index_lt == remove_fs, dim=1))]
        ad_collision_data.f_index_lt = ad_collision_data.f_index_lt[
            torch.where(~torch.all(ad_collision_data.f_index_lt == remove_fs, dim=1))]
    for remove_fs in remove_perm_gt:
        perm_collision_data.d_gt = perm_collision_data.d_gt[
            torch.where(~torch.all(perm_collision_data.f_index_gt == remove_fs, dim=1))]
        perm_collision_data.f_index_gt = perm_collision_data.f_index_gt[
            torch.where(~torch.all(perm_collision_data.f_index_gt == remove_fs, dim=1))]
    for remove_fs in remove_perm_lt:
        perm_collision_data.d_lt = perm_collision_data.d_lt[
            torch.where(~torch.all(perm_collision_data.f_index_lt == remove_fs, dim=1))]
        perm_collision_data.f_index_lt = perm_collision_data.f_index_lt[
            torch.where(~torch.all(perm_collision_data.f_index_lt == remove_fs, dim=1))]

    time1 = time.time() - time0
    print(f"overlap_solids calculations took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
    return ad_collision_data, perm_collision_data, full_mask
