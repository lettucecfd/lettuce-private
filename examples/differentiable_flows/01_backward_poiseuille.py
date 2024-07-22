import torch
import lettuce as lt
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from poiseuilleAD import PoiseuilleFlow2D

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--flow_name", default="Poiseuille", type=str, help="")
parser.add_argument("--default_device", default="cuda:0", type=str, help="")
args = vars(parser.parse_args())
flow_name = args["flow_name"]
default_device = args["default_device"]
torch.set_default_device(default_device)

"""Linear Function y = a*x"""
x = torch.ones(5, requires_grad=True)
x = x * 2  # input tensor
a = 2
y = torch.sum(a*x)
x.retain_grad()
y.backward()
print(x.grad)
print(a)

"""More complex Function y = a*x"""
x = torch.ones(5, requires_grad=True)  # input tensor
a = torch.randn(5)
b = torch.randn(5, 3)
y = torch.sum(a*x*x) + torch.sum(torch.matmul(x, b))
y.backward()
print(x.grad)


lattice = lt.Lattice(lt.D2Q9, device=default_device, use_native=False)
Ma = torch.ones(1, requires_grad=True) * 0.1
Re = torch.ones(1, requires_grad=True) * 100

flow = PoiseuilleFlow2D(
    resolution=20,
    reynolds_number=Re,
    mach_number=Ma,
    lattice=lattice
)
acceleration_lu = flow.units.convert_acceleration_to_lu(flow.acceleration)
force = lt.ShanChen(lattice, tau=flow.units.relaxation_parameter_lu,
                    acceleration=acceleration_lu)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu,
                            force=force)
x, y = flow.grid
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision,
                           streaming=streaming)

simulation.initialize_f_neq()


def getEnergyGradient(inputs, steps, name, plot=False):
    energy = torch.sum(lattice.incompressible_energy(simulation.f))
    inputs.retain_grad()
    energy.backward(retain_graph=True)
    print(f"Energy: {energy:.2e}, gradient of {name}: {inputs.grad.item():.2e}")
    if plot:
        u_x = flow.units.convert_velocity_to_pu(
                lattice.u(simulation.f)
            ).detach().cpu().numpy()[0].transpose()
        plt.imshow(u_x)
        plt.title(f"{flow_name} at it={steps}, energy: {energy:.2e}")
        plt.show()


steps, mlups = 0, 0
step_size = 100
getEnergyGradient(Ma, steps, "Mach number    ")
getEnergyGradient(Re, steps, "Reynolds number")
while steps < 1000:
    mlups += simulation.step(step_size)
    steps += step_size
    getEnergyGradient(Ma, steps, "Mach number    ", plot=True)
    getEnergyGradient(Re, steps, "Reynolds number")
# torch.sum(simulation.f).backward()



