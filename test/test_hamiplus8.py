import torch
import time
import dq_torch
from dqrobotics import DQ

a = torch.tensor([0, 2, 3, 4, 0, 6, 7, 8], dtype=torch.float32, device='cuda:0').repeat(1,1)
dq_robotics_a = DQ(0, 2, 3, 4, 0, 6, 7, 8)
print("dq_robotics_a: ", dq_robotics_a.hamiplus8())
print("torch_a: ", dq_torch.hamiplus8(a))