from examples.differentiable_geometry_REQUIRES_DLR_CODE.occ_ad import (
    OCC_has_ad, make_gp_Pnt, make_gp_Vec, torch_float_to_standard_adouble)
import torch
if OCC_has_ad:
    import OCC.Core.ForwardAD as ad
try:
    from OCC.Display.SimpleGui import init_display
    OCC_has_display = True
except:
    OCC_has_display = False

from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon,
                                     BRepBuilderAPI_MakeFace,
                                     BRepBuilderAPI_FindPlane,
                                     BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_MakeWire)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TColgp import TColgp_HArray1OfPnt
from OCC.Core.TColStd import TColStd_HArray1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import Geom_BSplineCurve


def build_terrain_2D(surface_data, depth=1.):
    """
    returns a TopoDS_Sold of a 2D-terrain model. surface_data is a 2D-array of sorted x and y coordinates representing
    a surface in 2D. This function connects the points to a polygonal curve, closes the curve to encompass a 2D face
    representing the internal area of the terrain and then extrudes the face in z-direction to obtain a solid.
    """
    stretch_grid = False
    z = -depth * 0.5

    # connect surface data to polygon
    polygon_builder = BRepBuilderAPI_MakePolygon()
    for x, y, *_ in surface_data:
        polygon_builder.Add(make_gp_Pnt(x.item(), y.item(), z))

    # close the polygon
    ymin = surface_data[:, 1].min().item()
    polygon_builder.Add(make_gp_Pnt(surface_data[-1, 0].item(), ymin - 10., z))
    polygon_builder.Add(make_gp_Pnt(surface_data[0, 0].item(), ymin - 10., z))
    polygon_builder.Close()
    polygon = polygon_builder.Wire()

    if not OCC_has_ad:
        searcher = BRepBuilderAPI_FindPlane(polygon, 1e-10)
    else:
        searcher = BRepBuilderAPI_FindPlane(polygon, ad.Standard_Adouble(1e-10))

    if searcher.Found():
        face_builder = BRepBuilderAPI_MakeFace(searcher.Plane(), polygon)
    else:
        face_builder = BRepBuilderAPI_MakeFace(polygon)
    surroundings = BRepPrimAPI_MakePrism(face_builder.Shape(), make_gp_Vec(0., 0., depth))
    return surroundings.Shape()


def build_house_from_points(surface_data, minz: float = 0., maxz: float = 1.):
    """
    returns a TopoDS_Sold of a 2D-house model. surface_data is a 2D-array of sorted x and y coordinates representing
    a surface in 2D. This function connects the points to a polygonal curve, closes the curve to encompass a 2D face
    representing the internal area of the terrain and then extrudes the face in z-direction to obtain a solid.
    """

    # connect surface data to polygon
    polygon_builder = BRepBuilderAPI_MakePolygon()
    for x, y in surface_data:
        polygon_builder.Add(make_gp_Pnt(x, y, minz))

    # close the polygon
    polygon_builder.Close()
    polygon = polygon_builder.Wire()

    if not OCC_has_ad:
        searcher = BRepBuilderAPI_FindPlane(polygon, 1e-10)
    else:
        searcher = BRepBuilderAPI_FindPlane(polygon, ad.Standard_Adouble(1e-10))

    if searcher.Found():
        face_builder = BRepBuilderAPI_MakeFace(searcher.Plane(), polygon)
    else:
        face_builder = BRepBuilderAPI_MakeFace(polygon)
    surroundings = BRepPrimAPI_MakePrism(face_builder.Shape(), make_gp_Vec(0., 0., maxz-minz))
    return surroundings.Shape()


def build_house(pos_x, pos_y, height, width, rel_roof_dx, rel_roof_dy, smooth=False, depth: float = 1., dim: int = 2,
                minz: float = None):
    """
    creates a 2D model of a house in the XY-plane and extrudes it by depth in the z-direction.

    The base shape of the house is rectangular and defined by the lower left corner (pos_x, pos_y) and its width and
    height. In addition, it has a roof, which is defined by rel_roof_dx and rel_roof_dy, where
     - rel_roof_dx defines the relative x position of the highest point of the roof with respect to the houses width
     - rel_roof_dy defines the relative y position of the highest point of the roof with respect to the houses height
     - smooth determines if the house gets a normal Giebeldach (smooth=False) or a smooth B-Spline curve of degree 2.

    For example rel_roof_dx = 0.5 and rel_roof_dy = 0.5 positions the "Giebel" at the center of the house at a height
    of (1+rel_roof_dy)*height.
    """

    # convert arguments in place to Standard_ADouble, if they aren't yet of that type
    pos_x, pos_y, height, width, rel_roof_dx, rel_roof_dy = [
        torch_float_to_standard_adouble(x) if isinstance(x, torch.Tensor) else ad.Standard_Adouble(x)
        if OCC_has_ad and not isinstance(x, ad.Standard_Adouble) 
        else x for x in [pos_x, pos_y, height, width, rel_roof_dx, rel_roof_dy]]

    minz = -depth * 0.5 if (minz is None or dim == 2) else minz

    # build the roof as a bspline:
    front_roof = make_gp_Pnt(pos_x, pos_y + height, minz)
    center_roof = make_gp_Pnt(pos_x + rel_roof_dx * width, pos_y + (1 + rel_roof_dy) * height, minz)
    back_roof = make_gp_Pnt(pos_x + width, pos_y + height, minz)

    degree = 1
    nknots = 5  # npoints + degree + 1
    mult = 1  # the multiplicity of each knot
    if smooth:
        degree = 2
        nknots = 2  # npoints + degree + 1 = 6, multiplicity start and end is 3
        mult = 3

    pnts = TColgp_HArray1OfPnt(1, 3)
    pnts.SetValue(1, front_roof)
    pnts.SetValue(2, center_roof)
    pnts.SetValue(3, back_roof)

    knots = TColStd_HArray1OfReal(1, nknots)
    for i in range(0, nknots):
        if not OCC_has_ad:
            knots.SetValue(i + 1, i)
        else:
            knots.SetValue(i + 1, ad.Standard_Adouble(i))

    mults = TColStd_Array1OfInteger(1, nknots)
    for i in range(1, nknots + 1):
        mults.SetValue(i, mult)

    curve = Geom_BSplineCurve(pnts, knots, mults, degree)
    roof = BRepBuilderAPI_MakeEdge(curve).Edge()

    back_house = make_gp_Pnt(pos_x + width, pos_y, minz)
    back = BRepBuilderAPI_MakeEdge(back_roof, back_house).Edge()

    front_house = make_gp_Pnt(pos_x, pos_y, minz)
    bottom = BRepBuilderAPI_MakeEdge(back_house, front_house).Edge()
    front = BRepBuilderAPI_MakeEdge(front_house, front_roof).Edge()

    wire_builder = BRepBuilderAPI_MakeWire()
    wire_builder.Add(roof)
    wire_builder.Add(back)
    wire_builder.Add(bottom)
    wire_builder.Add(front)
    house_wire = wire_builder.Wire()

    if not OCC_has_ad:
        searcher = BRepBuilderAPI_FindPlane(house_wire, 1e-10)
    else:
        searcher = BRepBuilderAPI_FindPlane(house_wire, ad.Standard_Adouble(1e-10))

    if searcher.Found():
        face_builder = BRepBuilderAPI_MakeFace(searcher.Plane(), house_wire)
    else:
        face_builder = BRepBuilderAPI_MakeFace(house_wire)

    house = BRepPrimAPI_MakePrism(face_builder.Face(), make_gp_Vec(0., 0., depth))

    return house.Shape()

