import os
import torch
import trimesh
from OCC.Core.TopoDS import TopoDS_Shape

from examples.occ_geometries.obstacleFunctions import (calculate_mask,
                                                       collect_collision_data)
from lettuce import Lattice
from lettuce.boundary import CollisionData


def getIBBdata(cad_data: trimesh.Trimesh or TopoDS_Shape,
               grid: tuple[torch.Tensor, ...], lattice: Lattice,
               no_store_coll: bool, res: float, dim: int, name: str,
               coll_data_path: str, only_mask: bool = False,
               redo_calculations: bool = False, parallel: bool = False,
               device: str = "cuda", cut_z: float = 0.,
               cluster: bool = False) -> CollisionData:
    if not os.path.exists(coll_data_path):
        os.mkdir(coll_data_path)
    coll_data_path = os.path.join(coll_data_path, f"{res:.2f}ppm_{dim}D")
    print(f"Looking for data for {name} in {coll_data_path}.")

    coll_data = CollisionData()

    mask_data_exists = (os.path.exists(
            os.path.join(coll_data_path, f"solid_mask_{name}.pt")
        ) and os.path.exists(
            os.path.join(coll_data_path, f"points_inside_{name}.pt")
        ))
    if mask_data_exists and not redo_calculations:
        print("Mask data found.")
        coll_data.solid_mask = torch.load(
                os.path.join(coll_data_path, f"solid_mask_{name}.pt"),
                map_location=device
            )
        coll_data.points_inside = torch.load(
                os.path.join(coll_data_path, f"points_inside_{name}.pt"),
                map_location=device
            )
    else:
        if not os.path.exists(coll_data_path):
            os.mkdir(coll_data_path)
        print("No mask data found or recalculation requested. "
              "Redoing mask calculations.")
        coll_data = calculate_mask(cad_data, grid, name=name,
                                   collision_data=coll_data, cut_z=cut_z)
        if not no_store_coll:
            torch.save(coll_data.solid_mask,
                       os.path.join(coll_data_path, f"solid_mask_{name}.pt"))
            torch.save(coll_data.points_inside,
                       os.path.join(coll_data_path,
                                    f"points_inside_{name}.pt"))
            print(f"Mask data saved to {coll_data_path}.")
    print(f"Mask data loaded for {name}.")

    if only_mask:
        return coll_data
    coll_data_exists = True
    for data_name in ['f_index_gt_', 'f_index_lt_', 'd_gt_', 'd_lt_',
                      'not_intersected_']:
        coll_data_exists *= os.path.exists(
                os.path.join(coll_data_path, f"{data_name}{name}.pt")
            )
    if coll_data_exists and not redo_calculations:
        print("Collision data found.")
        coll_data.f_index_gt = torch.load(
                os.path.join(coll_data_path, f"f_index_gt_{name}.pt"),
                map_location=device
            )
        coll_data.f_index_lt = torch.load(
                os.path.join(coll_data_path, f"f_index_lt_{name}.pt"),
                map_location=device
            )
        coll_data.d_gt = torch.load(
                os.path.join(coll_data_path, f"d_gt_{name}.pt"),
                map_location=device
            )
        coll_data.d_lt = torch.load(
                os.path.join(coll_data_path, f"d_lt_{name}.pt"),
                map_location=device
            )
        coll_data.not_intersected = torch.load(
                os.path.join(coll_data_path, f"not_intersected_{name}.pt"),
                map_location=device
            )
    else:
        print("No collision data found or recalculation requested. "
              "Redoing collision calculations.")
        coll_data = collect_collision_data(cad_data, coll_data, lattice, grid,
                                           name, outdir=coll_data_path,
                                           parallel=parallel, cut_z=cut_z,
                                           cluster=cluster)
        if not no_store_coll:
            torch.save(coll_data.f_index_gt,
                       os.path.join(coll_data_path, f"f_index_gt_{name}.pt"))
            torch.save(coll_data.f_index_lt,
                       os.path.join(coll_data_path, f"f_index_lt_{name}.pt"))
            torch.save(coll_data.d_gt,
                       os.path.join(coll_data_path, f"d_gt_{name}.pt"))
            torch.save(coll_data.d_lt,
                       os.path.join(coll_data_path, f"d_lt_{name}.pt"))
            torch.save(coll_data.not_intersected,
                       os.path.join(coll_data_path,
                                    f"not_intersected_{name}.pt"))
            print(f"Collision data saved to {coll_data_path}.")
    print(f"Collision data loaded for {name}.")

    return coll_data
