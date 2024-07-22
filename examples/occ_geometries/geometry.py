# add later: is_point_inside, project_point_on_surface (get inspiration from TiGL common functions' C++ implementation)
# add morde CAD functionality later
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Shell, TopoDS_Edge
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SHELL
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Compound
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.Precision import precision
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin, gp_Vec
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Sewing
from OCC.Core.GeomAPI import GeomAPI_IntCS
from OCC.Core.IntCurveSurface import IntCurveSurface_HInter
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopOpeBRep import TopOpeBRep_ShapeIntersector

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere

import torch

import warnings

try:
    from OCC.Display.SimpleGui import init_display
    print("Imported init_display from OCC.Display.SimpleGui, even if this is run on cluster. "
          "Display does not work on cluster.")
    OCC_has_display = True
except Exception as msg:
    OCC_has_display = False
    print(f"WARNING: No OCC display imported. Exception: {msg}")


def is_point_inside_solid(solid: TopoDS_Solid, point: gp_Pnt, bounding_box=None):
    """checks if a point is contained in a TopoDS_Solid.

    :param solid: A solid representing a 3d geometry
    :type solid: TopoDS_Solid
    :param point: The point
    :type point: gp_Pnt
    :return: True, if the point is contained in the face and False otherwise
    :rtype: bool
    """
    tol = precision.Confusion()

    if bounding_box is None:
        bounding_box = Bnd_Box()
        brepbndlib.Add(solid, bounding_box)

    if bounding_box.IsOut(point):
        return False

    solidClassifier = BRepClass3d_SolidClassifier(solid)
    solidClassifier.Perform(point, tol)
    state = solidClassifier.State()

    if state == TopAbs_ON or state == TopAbs_IN:
        return True
    return False


def intersect_boundary_with_ray(fluid_point: gp_Pnt, solid_point: gp_Pnt, faces: TopoDS_Shell or TopoDS_Compound,
                                cluster: bool = False):
    # warning: if the ray does not cut the shape, the solid point will be returned
    vector = gp_Vec(fluid_point, solid_point)
    direction = gp_Dir(vector)
    # ray = gp_Lin(fluid_point, direction)
    edg = BRepBuilderAPI_MakeEdge(fluid_point, solid_point).Edge()
    extrema_distShapeShape = BRepExtrema_DistShapeShape(edg, faces)
    # find the number of points with a minimum distance. This should be at least one
    nPoints = extrema_distShapeShape.NbSolution()
    if nPoints == 0:
        raise RuntimeError("Projection of fluid_point to shape failed.")

    pnt = gp_Pnt()
    positive_cut = False  # using dot product do determine that minimum distance was found in direction of ray
    d = 1.
    dist_set = False  # tracking if a d <=1. was found
    dist_fluid_solid = fluid_point.Distance(solid_point)

    if nPoints >= 1:

        for i in range(1, nPoints + 1):
            point_i = extrema_distShapeShape.PointOnShape1(i)
            vec_i = gp_Vec(fluid_point, point_i)
            try:
                direction_intersection = gp_Dir(vec_i)
            except Exception as msg:
                print(f"WARNING: gp_Dir(vec_i)-Exception. Problably, ray between "
                      f"fluid point at {[round(_, 2) for _ in fluid_point.Coord()]} "
                      f"and solid point at {[round(_, 2) for _ in solid_point.Coord()]} "
                      f"does not intersect boundary: {msg}.")
                display_ray(faces, fluid_point, solid_point, nPoints, extrema_distShapeShape, edg, cluster)
                continue
            along_ray = direction_intersection.Dot(direction)

            if along_ray >= 0:
                positive_cut = True

                d_i = point_i.Distance(fluid_point) / dist_fluid_solid

                if d_i <= d:
                    d = d_i
                    pnt = point_i
                    dist_set = True

    if not positive_cut or not dist_set:
        print(f"WARNING: Ray from {fluid_point.Coord()} to {solid_point.Coord()} "
              f"intersects only at {point_i.Coord()} with d={d:.2f}, which must be in [0, 1]. Setting d to 1.")
        pnt = solid_point
        d = 1.
        display_ray(faces, fluid_point, solid_point, nPoints, extrema_distShapeShape, edg, cluster)
        # raise RuntimeError("No valid intersection in given direction. Projection of point to shape failed.")
    return pnt, d  # TODO: get dual values of pnt.Coord() and d


def display_ray(shape: TopoDS_Shape = None, fluid_point: gp_Pnt = None, solid_point: gp_Pnt = None, nPoints: int = 0,
                extrema_distShapeShape: BRepExtrema_DistShapeShape = None, edg: TopoDS_Edge = None,
                cluster: bool = False):
    if OCC_has_display and not cluster:
        print("OCC_has_display is True. Blue: Fluid Point. Green: Solid Point. Red: Intersection Point.")
        display, start_display, add_menu, add_function_to_menu = init_display()
        if shape is not None:
            display.DisplayShape(shape, transparency=0.5)
        if fluid_point is not None:
            display.DisplayShape(fluid_point, color='Blue')
        if solid_point is not None:
            display.DisplayShape(solid_point, color='Green')
        if nPoints > 0 and extrema_distShapeShape is not None:
            for i in range(1, nPoints + 1):
                display.DisplayShape(extrema_distShapeShape.PointOnShape1(i), color='Red')
        display.FitAll()
        if edg is not None:
            display.DisplayShape(edg, color='Red')
        start_display()
    else:
        print("No OCC_has_display or running on cluster. Skipping display_ray.")
    return


def extract_faces(solid: TopoDS_Solid or TopoDS_Shape) -> TopoDS_Shape:
    faces = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(faces)
    faceExplorer = TopExp_Explorer(solid, TopAbs_SHELL)
    while faceExplorer.More():
        face = faceExplorer.Current()
        builder.Add(faces, face)
        faceExplorer.Next()
    return faces

