import torch
import time
import dq_torch
from dqrobotics import DQ

a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0, 0, 0], dtype=torch.float32, device='cuda:0').repeat(1,1)
dq_robotics_a = DQ(0.5, 0.5, 0.5, 0.5, 0.0, 0, 0,0)
print("dq_robotics_a: ", dq_robotics_a.rotation_axis())
print("torch_a: ", dq_torch.rotation_axis(a))
