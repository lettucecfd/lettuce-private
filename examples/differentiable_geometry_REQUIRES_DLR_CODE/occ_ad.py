"""helper functions when working with AD-enabled pythonocc
"""
import torch
import torch.autograd.forward_ad as fwAD
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

try:
    import OCC.Core.ForwardAD as ad

    OCC_has_ad = True
except:
    OCC_has_ad = False
from OCC.Core.gp import gp_Pnt, gp_Vec


def torch_float_to_standard_adouble(x: torch.Tensor):
    assert len(x.shape) == 0  # this should be only for 0D tensors,
    # i.e. torch floats
    if OCC_has_ad:
        primal = fwAD.unpack_dual(x).primal
        tangent = fwAD.unpack_dual(x).tangent
        if tangent is None:
            tangent = 0.
        ret = ad.Standard_Adouble(primal.item())
        ret.setADValue(0, float(tangent))
        return ret
    else:
        return x.item()


def standard_adouble_to_torch_float(x):
    if OCC_has_ad and isinstance(x, ad.Standard_Adouble):
        return fwAD.make_dual(torch.tensor(x.getValue()),
                              torch.tensor(x.getADValue(0)))
    else:
        return fwAD.make_dual(torch.tensor(x), torch.tensor(0.1234))


def torch_tensor_to_standard_adouble_list(x: torch.Tensor):
    assert len(x.shape) == 1  # this should be only for 1D tensors
    if OCC_has_ad:
        adouble_list = []
        for a in x:
            adouble_list.append(torch_float_to_standard_adouble(a))
        return
    else:
        return x.tolist()


def standard_adouble_list_to_torch_tensor(x):
    result = []
    for a in x:
        result.append(standard_adouble_to_torch_float(a))
    return fwAD.make_dual(torch.tensor(result),
                          torch.tensor([fwAD.unpack_dual(_).tangent for _ in
                                        result]))


def make_gp_Pnt(x: float, y: float, z: float):
    """creates a gp_Pnt out of three coordinates. Wraps the coordinates in
    ad.adouble, if AD is enabled, i.e. if OCC_has_ad is ture

    :param x: x-value of the point
    :type x: float
    :param y: y-value of the point
    :type y: float
    :param z: z-value of the point
    :type z: float
    :return: _description_
    :rtype: _type_
    """
    if OCC_has_ad:
        return gp_Pnt(ad.Standard_Adouble(x), ad.Standard_Adouble(y),
                      ad.Standard_Adouble(z))
    else:
        return gp_Pnt(x, y, z)


def make_gp_Vec(x: float, y: float, z: float):
    """creates a gp_Pnt out of three coordinates. Wraps the coordinates in
    ad.adouble, if AD is enabled, i.e. if OCC_has_ad is ture

    :param x: x-value of the point
    :type x: float
    :param y: y-value of the point
    :type y: float
    :param z: z-value of the point
    :type z: float
    :return: _description_
    :rtype: _type_
    """
    if OCC_has_ad:
        return gp_Vec(ad.Standard_Adouble(x), ad.Standard_Adouble(y),
                      ad.Standard_Adouble(z))
    else:
        return gp_Vec(x, y, z)

def make_BRepPrimAPI_MakeBox(x: float, y: float, z: float):
    """creates a gp_Pnt out of three coordinates. Wraps the coordinates in
    ad.adouble, if AD is enabled, i.e. if OCC_has_ad is ture

    :param x: x-value of the point
    :type x: float
    :param y: y-value of the point
    :type y: float
    :param z: z-value of the point
    :type z: float
    :return: _description_
    :rtype: _type_
    """
    if OCC_has_ad:
        return BRepPrimAPI_MakeBox(ad.Standard_Adouble(x),
                                   ad.Standard_Adouble(y),
                                   ad.Standard_Adouble(z))
    else:
        return BRepPrimAPI_MakeBox(x, y, z)


