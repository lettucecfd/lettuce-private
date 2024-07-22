import torch
from obstacleCylinderAD import ObstacleCylinder
import lettuce as lt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--default_device", default="cuda", type=str,
                    help="")
torch.autograd.set_detect_anomaly(True)
args = vars(parser.parse_args())
default_device = args["default_device"]
torch.set_default_device(default_device)

lattice = lt.Lattice(lt.D2Q9, device=default_device, use_native=False)
Ma = torch.ones(1, requires_grad=True) * 0.1
Re = torch.ones(1, requires_grad=True) * 100
radius = torch.ones(1, requires_grad=True)
res = 6  # grid points for the cylinder
flow = ObstacleCylinder(
    shape=(6*res, 4*res),
    reynolds_number=Re,
    mach_number=Ma,
    lattice=lattice,
    char_length_pu=res,
    char_length_lu=res,
    radius=radius
)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)

x, y = flow.grid
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision,
                           streaming=streaming)
simulation.initialize_f_neq()

mlups = simulation.step(100)

energy = torch.sum(lattice.incompressible_energy(simulation.f))
print(f"Total energy: {energy:.2e}")

Ma.retain_grad()
Re.retain_grad()
radius.retain_grad()
energy.backward()
print(f"Gradient of Ma: {lattice.convert_to_numpy(Ma.grad)[0]:.2e}")
print(f"Gradient of Re: {lattice.convert_to_numpy(Re.grad)[0]:.2e}")
print(f"Gradient of radius: {lattice.convert_to_numpy(radius.grad)[0]:.2e}")
