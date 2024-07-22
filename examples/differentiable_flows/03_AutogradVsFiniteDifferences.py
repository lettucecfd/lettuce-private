import torch
from poiseuilleAD import PoiseuilleFlow2D
import lettuce as lt
from time import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--default_device", default="cuda", type=str, help="")
args = vars(parser.parse_args())

default_device = args["default_device"]
torch.set_default_device(default_device)

print("Manual calculation of gradient of energy over Ma...")
time1 = time()
energy = [0, 0]
res = 100
Ma = 0.1
Re = 500
nmax = 1000
Ma_list = [Ma*(1-1e-4), Ma*(1+1e-4)]

def get_simulation(Ma):
    lattice = lt.Lattice(lt.D2Q9, device=default_device, use_native=False)
    flow = PoiseuilleFlow2D(res, Re, Ma, lattice)
    acceleration_lu = flow.units.convert_acceleration_to_lu(flow.acceleration)
    force = lt.ShanChen(lattice, tau=flow.units.relaxation_parameter_lu,
                        acceleration=acceleration_lu)
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu,
                                force=force)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision,
                               streaming=streaming)
    simulation.initialize_f_neq()
    return simulation, lattice

for i in range(2):
    simulation, lattice = get_simulation(Ma_list[i])
    simulation.step(nmax)
    energy[i] = torch.sum(lattice.incompressible_energy(simulation.f))

Ma_grad = ((energy[1] - energy[0])
           / (Ma_list[1] - Ma_list[0]))
print(f"Energy: {.5*(energy[1]+energy[0]):.2e}, manual gradient of Ma: {Ma_grad:.2e} took "
      f"{time() - time1:.2f} s.")

print("\nNow calculating with torch autograd...")
time1 = time()
Ma = torch.ones(1, requires_grad=True) * Ma

simulation, lattice = get_simulation(Ma)
simulation.step(nmax)

energy = torch.sum(lattice.incompressible_energy(simulation.f))
Ma.retain_grad()
energy.backward(retain_graph=True)
print(f"Energy: {energy:.2e}, "
      f"autogradient of Ma: {lattice.convert_to_numpy(Ma.grad)[0]:.2e} "
      f"took {time() - time1:.2f} s.")
