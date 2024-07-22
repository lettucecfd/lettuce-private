import os
from math import ceil

import trimesh

from ad_surface.geometric_building_model import build_terrain_2D, build_house
from ad_surface.helperFunctions.clean_surface_data import get_clean_surface_data


def getInputData(datatype: str, dim: int, res: float, interpolateres: float = 1., interpolatecsv: bool = False,
                 depth: float = 1., debug: bool = False):
    csv = True if 'csv' in datatype else False
    if csv:
        while True:
            stl = str(input("WARNING: Reading csv data is very inefficient, especially in 3D. "
                            "Do you want to use the stl file instead? [y/n]"))
            if stl.lower() not in ('y', 'n'):
                print("Sorry, I didn't understand that.")
                continue
            else:
                break
        if stl.lower() == 'n':
            if '2d' in datatype:
                dim = 2
            else:
                dim = 3
            filename = f"nsg_study_landscape{'_3D' if dim == 3 else ''}.csv"
            surface_csv, dim, dim_data = get_clean_surface_data(
                os.path.join(os.getcwd(), f"data/{filename}"),
                res=res, debug=debug, dim=dim, interpolate=interpolatecsv,
                depth=depth, res_interp=interpolateres)
            terrain_builder = build_terrain_2D
            landscape = terrain_builder(surface_csv, depth=depth)
            minx, maxx = [[surface_csv[:, _].min().item() for _ in range(dim_data)],
                          [surface_csv[:, _].max().item() for _ in range(dim_data)]]
            minx[1] -= 1 / res  # making sure lowest surface point is within grid (by half grid point)
        else:
            csv = False
    if not csv:
        dim = dim if dim is not None else 3
        filename = "landscape_and_existing_buildings_remeshed.stl"
        landscape = trimesh.load_mesh('data/' + filename)
        bounds = landscape.bounds
        minx, maxx = [[bounds[0, 0], bounds[0, 2], bounds[0, 1]], [bounds[1, 0], bounds[1, 2], bounds[1, 1]]]
        if dim == 2:
            minx, maxx = minx[:2], maxx[:2]
        minx[1] += 2  # making sure bottom of solid is outside of grid (by half grid point) to have no IBB there

    scale_y = 8  # height is 5 times the surface height
    minx[0] += 5  # making sure left side of solid is within grid
    maxx[0] -= 5  # making sure right side of solid is within grid
    maxx[1] *= scale_y
    nx = ceil((maxx[0] - minx[0]) * res)  # downstream direction
    ny = ceil((maxx[1] - minx[1]) * res)
    shape = (nx, ny)
    if dim == 3:
        minx[2] += 5
        maxx[2] -= 5
        nz = ceil((maxx[2] - minx[2]) * res)  # depth (i.e., 'optional' dimension)
        shape = (nx, ny, nz)
    domain_constraints = (minx, maxx)

    dx = (domain_constraints[1][0] - domain_constraints[0][0]) / (shape[0] - 1)
    dy = (domain_constraints[1][1] - domain_constraints[0][1]) / (shape[1] - 1)
    dz = (domain_constraints[1][2] - domain_constraints[0][2]) / (shape[2] - 1) if dim == 3 else 0

    print(
        f"Domain goes from {[round(_, 2) for _ in minx]} to {[round(_, 2) for _ in maxx]}, resolved on {shape} points.")
    print(f"This results in a resolution of {dx:.4f} x {dy:.4f}{' x ' + str(round(dz, 4)) if dim == 3 else ''} meters.")
    print(f"Grid is distorted by {abs(1 - dx / dy):.3%} in x-y plane.")
    if dim == 3:
        print(f"Grid is distorted by {abs(1 - dy / dz):.3%} in z-y plane and {abs(1 - dx / dz):.3%} in x-z plane.")

    return landscape, dim, domain_constraints, shape


def getHouse(house_name: str, minz: float = 63, maxz: float = 123):
    if 'house' in house_name:
        if house_name in ['house', 'testhouse']:
            house_data = build_house(pos_x=100, pos_y=5, height=9, width=20, rel_roof_dx=0., rel_roof_dy=0.5,
                                     smooth=True)
        elif house_name == 'house3d':
            house_data = build_house(pos_x=100, pos_y=5, height=9, width=20, rel_roof_dx=0., rel_roof_dy=0.5,
                                     smooth=True, minz=minz, depth=maxz - minz)
        elif 'house_upwards_wedge' in house_name:
            house_data = build_house(pos_x=100, pos_y=5, height=10, width=20, rel_roof_dx=1., rel_roof_dy=0.5,
                                     smooth=False)
        elif 'house_downwards_wedge' in house_name:
            house_data = build_house(pos_x=100, pos_y=5, height=10, width=20, rel_roof_dx=0., rel_roof_dy=0.5,
                                     smooth=False)
        elif 'house_flat' in house_name:
            house_data = build_house(pos_x=100, pos_y=5, height=10, width=20, rel_roof_dx=0., rel_roof_dy=0.,
                                     smooth=False)
        elif 'house_pointy' in house_name:
            house_data = build_house(pos_x=100, pos_y=5, height=10, width=20, rel_roof_dx=0.5, rel_roof_dy=0.5,
                                     smooth=False)
        else:
            raise ValueError(f"Unknown house_name: {house_name}")
    else:
        raise ValueError(f"Unknown house_name: {house_name}")
    return house_data
