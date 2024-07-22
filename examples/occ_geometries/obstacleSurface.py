import time
from math import floor

import numpy as np
import torch
import trimesh
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Shell, TopoDS_Compound
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse

from examples.occ_geometries.obstacleFunctions import overlap_solids, collect_collision_data, calculate_mask, makeGrid
from lettuce.boundary import (EquilibriumBoundaryPU, InterpolatedBounceBackBoundary, EquilibriumOutletP,
                              BounceBackBoundary, PartiallySaturatedBoundary, LettuceBoundary, CollisionData)
from lettuce.lattices import Lattice
from lettuce.unit import UnitConversion
from lettuce.util import append_axes


class BoundaryObject:
    collision_data: CollisionData
    ad_enabled: bool
    shell: TopoDS_Compound or TopoDS_Shell
    boundary: LettuceBoundary

    def __init__(self, occ_object: TopoDS_Solid or torch.Tensor or trimesh.Trimesh,
                 boundary_type: InterpolatedBounceBackBoundary or BounceBackBoundary or PartiallySaturatedBoundary,
                 grid: tuple[torch.Tensor, ...], lattice: Lattice,
                 ad_enabled: bool = False, saturation: float = None, tau: float = None, name: str = None,
                 parallel=False, cut_z: float = 0, cluster: bool = False):
        self.shell = None
        self.occ_object = occ_object
        self.boundary_type = boundary_type
        self.ad_enabled = ad_enabled
        self.name = name
        self.collision_data_known = False
        self.points_inside_known = False
        self.unique_boundary = True
        self.grid = grid
        self.lattice = lattice
        self.cut_z = cut_z
        self.cluster = cluster
        self.collision_data = CollisionData()
        if self.boundary_type is PartiallySaturatedBoundary:
            assert saturation is not None and tau is not None
            self.saturation = saturation
            self.tau = tau
        if isinstance(occ_object, torch.Tensor):
            assert self.boundary_type is BounceBackBoundary or self.boundary_type is PartiallySaturatedBoundary
            self.collision_data.solid_mask = occ_object
            self.points_inside_known = True
        if isinstance(occ_object, CollisionData):
            self.collision_data = occ_object
            if hasattr(self.collision_data, 'solid_mask'):
                self.points_inside_known = True
            if hasattr(self.collision_data, 'd_gt'):
                self.collision_data_known = True
        self.is_occ = type(self.occ_object) is TopoDS_Shape or type(self.occ_object) is TopoDS_Solid
        self.is_tri = type(self.occ_object) is trimesh.Trimesh
        assert not (self.is_tri and self.is_occ)
        self.parallel = parallel
        return

    def get_boundary(self):
        print(f"get_boundary for '{self.name}' of type {self.boundary_type}")
        if isinstance(self.occ_object, torch.Tensor):
            if self.boundary_type is BounceBackBoundary:
                self.boundary = BounceBackBoundary(self.solid_mask, self.lattice)
            elif self.boundary_type is PartiallySaturatedBoundary:
                self.boundary = PartiallySaturatedBoundary(self.solid_mask, self.lattice,
                                                           tau=self.tau, saturation=self.saturation)
        else:
            if self.boundary_type is InterpolatedBounceBackBoundary:
                if not hasattr(self.collision_data, 'f_index_lt'):
                    self.collect_collision_data()
                self.boundary = InterpolatedBounceBackBoundary(self.solid_mask, self.lattice,
                                                               collision_data=self.collision_data,
                                                               ad_enabled=self.ad_enabled)
            elif self.boundary_type is BounceBackBoundary:
                self.boundary = BounceBackBoundary(self.solid_mask, self.lattice)
            elif self.boundary_type is PartiallySaturatedBoundary:
                self.boundary = PartiallySaturatedBoundary(self.solid_mask, self.lattice,
                                                           tau=self.tau, saturation=self.saturation)
            else:
                raise ValueError('Boundary type not supported')
        return self.boundary

    @property
    def solid_mask(self):
        if not hasattr(self.collision_data, 'solid_mask'):
            self.calculate_points_inside()
        return self.collision_data.solid_mask

    def calculate_points_inside(self):
        print(f"calculate_points_inside for '{self.name}' of type {self.boundary_type}")
        if self.points_inside_known:
            print("WARNING: calculate_points_inside has already been executed! Jumping this.\n")
            return
        self.collision_data = calculate_mask(self.occ_object, self.grid, self.name, self.collision_data)
        self.points_inside_known = True
        return

    def collect_collision_data(self):
        if self.boundary_type is not InterpolatedBounceBackBoundary:
            print("WARNING: Collecting collision data, but boundary type does not require it.")
        if self.collision_data_known:
            print("WARNING: Collision data already collected! Jumping this.")
            return
        if not self.points_inside_known:
            print("WARNING: points_inside not known yet. Doing calculate_points_inside now.")
            self.calculate_points_inside()
        self.collision_data = collect_collision_data(self.occ_object, self.collision_data, self.lattice, self.grid,
                                                     self.name, parallel=self.parallel, cut_z=self.cut_z,
                                                     cluster=self.cluster)
        self.collision_data_known = True
        return


class ObstacleSurface:
    boundary_objects: list[BoundaryObject] = []
    overlaps_must_be_calculated: bool = False
    y_interp: torch.Tensor = None
    ux: torch.Tensor = None
    shape: tuple[int, int, int] or tuple[int, int]

    def __init__(self, res: float, reynolds_number: float, mach_number: float, lattice: Lattice,
                 domain_constraints: tuple, shape: tuple, depth: float,
                 char_length_pu: float = 1, char_velocity_pu=1, u_init: 0 or 1 or 2 = 0,
                 debug=False, fwbb: bool = False, house=False, parallel=False, cluster=False):
        # flow and boundary settings
        self.u_init = u_init  # toggle: initial solution velocity profile type
        self.lattice = lattice
        assert len(shape) == lattice.D
        self.shape = shape
        self.ndim = lattice.D
        self.char_length_pu = char_length_pu  # characteristic length
        self.res = res
        self.all_fwbb = fwbb  # setting all boundaries to full way bounce back
        self.debug = debug
        self.house = house
        self.depth = depth
        self.domain_constraints = domain_constraints
        self.boundaries_are_initialized = False
        self.cluster = cluster

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=res,
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu  # reminder: u_char_lu = Ma * cs_lu = Ma * 1/sqrt(3)
        )
        self.parallel = parallel

    def add_boundary(self, boundary_object: TopoDS_Solid or torch.Tensor or CollisionData,
                     boundary_type: InterpolatedBounceBackBoundary or BounceBackBoundary or PartiallySaturatedBoundary,
                     ad_enabled: bool = False, saturation: float = None, tau: float = None, name: str = None,
                     cut_z: float = 0, cluster: bool = False):
        self.boundary_objects.append(
            BoundaryObject(boundary_object, boundary_type, self.grid, self.lattice, ad_enabled, saturation,
                           tau, name, self.parallel, cut_z=cut_z, cluster=cluster))
        return

    @property
    def solid_mask(self):
        if not hasattr(self, '_solid_mask'):
            self.overlap_all_solid_masks()
        return self._solid_mask

    def initial_solution(self, x: torch.Tensor):
        # initial velocity field: "u_init"-parameter
        p = torch.zeros_like(x[0], dtype=self.lattice.dtype)[None, ...]
        y = self.grid[1]
        self.initialize_object_boundaries()
        # if not hasattr(self, 'solid_mask'):
        #     # if not all(hasattr(_, 'solid_mask') for _ in )
        #     self.overlap_all_solid_masks()
        if self.u_init == 0:  # 0: uniform u=0
            u = torch.zeros_like(torch.stack(x), dtype=self.lattice.dtype)
            u_max_pu = self.units.characteristic_velocity_pu * torch.eye(self.ndim)[0]
            u_max_pu = append_axes(u_max_pu, self.ndim)
            self.ux = ~self.solid_mask * u_max_pu
            return p, u
        elif self.u_init == 1:  # 1: uniform u=1 is handled by using self.ux in the stack below
            u_max_pu = self.units.characteristic_velocity_pu
            self.ux = ~self.solid_mask * u_max_pu
        elif self.u_init == 2:  # 2: free-flow velocity profile according to
            # https://www.simscale.com/knowledge-base/atmospheric-boundary-layer-abl/.
            K = 0.4
            y0 = 2  # otherwise referred to as z0, meaning height
            # y0 should be expected around 1 according to descriptions, but in long-term the profile is much slower
            u_ref = 0.99
            H_ref = 2
            u_dash = K * u_ref / np.log((H_ref + y0) / y0)
            h_cartesian = y
            y_solid = torch.max(torch.where(self.solid_mask, y, y.min() - 1), dim=1)[0]
            h = h_cartesian - y_solid[:, None]
            self.ux = u_dash / K * torch.log((torch.where(h > 0, h, 0) + y0) / y0)
        else:
            raise NotImplementedError("Specify u_init = 0, 1, or 2")
        if self.ndim == 3:
            u = torch.stack((self.ux, torch.zeros_like(y), torch.zeros_like(y)), 0)
        else:
            u = torch.stack((self.ux, torch.zeros_like(y)), 0)
        return p, u.to(self.lattice.dtype)

    @property
    def grid(self):
        return makeGrid(self.domain_constraints, self.shape)
        # minx, maxx = self.domain_constraints
        # xyz = tuple(self.units.convert_length_to_pu(torch.linspace(minx[_], maxx[_], self.shape[_]))
        #             for _ in range(self.ndim))  # tuple of lists of x,y,(z)-values/indices
        # return torch.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (und z-)values/indices

    @property
    def boundaries(self):
        print("Doing boundaries")
        time0 = time.time()
        x, y = self.grid[:2]
        if self.u_init == 0:
            u_in = self.units.characteristic_velocity_pu * torch.eye(self.ndim)[0]
            u_top = u_in
        else:
            self.initial_solution(self.grid[0])
            ux_in = self.ux[0, :, :][None, :, :] if self.ndim == 3 else self.ux[0, :][None, :]
            u_in = torch.stack((ux_in, torch.zeros_like(ux_in), torch.zeros_like(ux_in)), 0) \
                if self.ndim == 3 else torch.stack((ux_in, torch.zeros_like(ux_in)), 0)
            ux_top = self.ux[:, -1, :][:, None, :] if self.ndim == 3 else self.ux[:, -1][:, None]
            u_top = torch.stack((ux_top, torch.zeros_like(ux_top), torch.zeros_like(ux_top)), 0) \
                if self.ndim == 3 else torch.stack((ux_top, torch.zeros_like(ux_top)), 0)
        outlet_direction = [1, 0] if self.ndim == 2 else [1, 0, 0]
        outlet_boundary = EquilibriumOutletP(self.units.lattice, outlet_direction)  # outlet in positive x-direction

        top_boundary = EquilibriumBoundaryPU(  # outlet
            y >= y.max(), self.units.lattice, self.units, u_top
        )
        boundaries = [outlet_boundary, top_boundary]
        if self.all_fwbb:
            for obstacle in self.boundary_objects:
                boundaries.append(BounceBackBoundary(obstacle.collision_data.solid_mask, self.units.lattice))
        else:
            # adding all boundaries
            for obj in [_ for _ in self.boundary_objects if _.unique_boundary]:
                boundaries.append(obj.get_boundary())
        # if not hasattr(self, 'solid_mask'):
        #     self.overlap_all_solid_masks()
        boundaries.append(EquilibriumBoundaryPU(  # inlet
            (x <= x.min()) * (~self.solid_mask), self.units.lattice, self.units, u_in
        ))
        time1 = time.time() - time0
        print(f"boundaries took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
        return boundaries

    def initialize_object_boundaries(self):
        if self.boundaries_are_initialized:
            return
        if not self.all_fwbb:
            for boundary_type in [InterpolatedBounceBackBoundary, PartiallySaturatedBoundary, BounceBackBoundary]:
                boundary_list = [obj for obj in self.boundary_objects
                                 if obj.boundary_type is boundary_type
                                 and obj.occ_object is not None
                                 and obj.unique_boundary]
                if boundary_list:
                    if boundary_type is InterpolatedBounceBackBoundary:
                        ad_list = [obj for obj in boundary_list if obj.ad_enabled and obj.unique_boundary]
                        perm_list = [obj for obj in boundary_list if not obj.ad_enabled and obj.unique_boundary]
                        if not ad_list or not perm_list:
                            boundary_list = ad_list if ad_list else perm_list
                        if len(boundary_list) == 1:  # only one solid with ibb boundary
                            ibb_boundary = boundary_list[0]
                            ibb_boundary.collect_collision_data()
                            if ibb_boundary.ad_enabled:
                                self.ibb_ad_boundary = ibb_boundary
                            else:
                                self.ibb_perm_boundary = ibb_boundary
                        elif not ad_list or not perm_list:  # all objects are ad or permanent -> no overlap situation -> combine all solids
                            ibb_boundary = self.combine_boundary_objects(boundary_list)
                            ibb_boundary.collect_collision_data()
                            if ibb_boundary.ad_enabled:
                                self.ibb_ad_boundary = ibb_boundary
                            else:
                                self.ibb_perm_boundary = ibb_boundary
                        else:
                            ad_boundary = self.combine_boundary_objects(ad_list) \
                                if len(ad_list) > 1 else ad_list[0]
                            ad_boundary.calculate_points_inside()
                            perm_boundary = self.combine_boundary_objects(perm_list) \
                                if len(perm_list) > 1 else perm_list[0]
                            perm_boundary.calculate_points_inside()
                            ad_boundary.collision_data, perm_boundary.collision_data, full_mask = overlap_solids(
                                ad_boundary.collision_data,
                                perm_boundary.collision_data)
                            self.ibb_ad_boundary = ad_boundary
                            self.ibb_perm_boundary = perm_boundary
                    else:
                        occ_boundary_list = [_ for _ in boundary_list if _.is_occ]
                        if len(occ_boundary_list) > 1:
                            occ_boundary = self.combine_boundary_objects(occ_boundary_list)
                            occ_boundary.calculate_points_inside()
                        if boundary_type is PartiallySaturatedBoundary:
                            self.partial_mask = torch.zeros_like(boundary_list[0].solid_mask)
                            for boundary in boundary_list:
                                self.partial_mask = torch.logical_or(self.partial_mask, boundary.solid_mask)
        self.boundaries_are_initialized = True
        return

    def overlap_all_solid_masks(self):
        print("overlap_all_solid_masks")
        self.initialize_object_boundaries()
        time0 = time.time()
        assert self.boundary_objects is not None
        boundaries_list = [_ for _ in self.boundary_objects
                           if _.unique_boundary and _.boundary_type is not PartiallySaturatedBoundary]
        # for boundary in boundaries_list:
        #     if not hasattr(boundary.collision_data, 'solid_mask'):
        #         boundary.calculate_points_inside()
        self._solid_mask = torch.zeros_like(boundaries_list[0].solid_mask, dtype=torch.bool)
        for boundary_object in boundaries_list:
            self._solid_mask = self.solid_mask | boundary_object.solid_mask
        time1 = time.time() - time0
        print(f"overlap_all_solid_masks took {floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].\n")
        return

    def combine_boundary_objects(self, boundary_objects: list[BoundaryObject]) -> BoundaryObject:
        time0 = time.time()
        compound = boundary_objects[0].occ_object
        for obj in boundary_objects:
            compound = BRepAlgoAPI_Fuse(compound, obj.occ_object).Shape()
            obj.unique_boundary = False
        solid = compound
        combined_boundary = BoundaryObject(solid, InterpolatedBounceBackBoundary, self.grid, self.lattice,
                                           boundary_objects[0].ad_enabled,
                                           name=f"{[_.name + '_' for _ in boundary_objects]}", parallel=self.parallel,
                                           cluster=self.cluster)
        self.boundary_objects.append(combined_boundary)
        time1 = time.time() - time0
        print(f"combine_boundary_objects '{[_.name for _ in boundary_objects]}' took "
              f"{floor(time1 / 60):02d}:{floor(time1 % 60):02d} [mm:ss].")
        return combined_boundary

    def overlap_ibb_solids(self, ad_boundary: BoundaryObject, perm_boundary: BoundaryObject):
        for boundary in [ad_boundary, perm_boundary]:
            boundary.calculate_points_inside()
            boundary.collect_collision_data()
        print("Doing overlap_ibb_solids")
        ad_boundary.collision_data, perm_boundary.collision_data, full_mask = overlap_solids(ad_boundary.collision_data,
                                                                                             perm_boundary.collision_data)
        self._solid_mask = torch.logical_or(full_mask, self._solid_mask)
        return

    def _unit_vector(self, i=0):
        return torch.eye(self.ndim)[i]
