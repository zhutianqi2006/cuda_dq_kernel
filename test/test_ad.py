import torch
import time
import dq_torch
from dqrobotics import DQ

a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.3, 0.2], dtype=torch.float32, device='cuda:0').repeat(1,1)
b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda:0').repeat(1,1)
dq_robotics_a = DQ(0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.3, 0.2)
dq_robotics_b = DQ(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
print("dq_robotics_a: ", dq_robotics_a.Ad(dq_robotics_b))
print("torch_a: ", dq_torch.Ad(a,b))
