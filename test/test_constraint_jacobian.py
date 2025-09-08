import torch
from dq_torch import  norm, haminus8, conj,dq_mult, rel_abs_pose_rel_jac, rel_dir_rect, project_q_by_constraint_jacobian
import numpy as np
import cProfile
from dqrobotics.robot_modeling import DQ_SerialManipulatorMDH, DQ_CooperativeDualTaskSpace
from dqrobotics import DQ
import time

##########################################################
############### get rel nullspace define #################
##########################################################   
# cpu robot1
@torch.jit.script
def get_rel_jacobian_null(jac_batch: torch.Tensor):
    # 形状：jac_batch [B, m, n]，n = robot1_q_num + robot2_q_num
    B, m, n = jac_batch.shape  # 运行时检查可保留/删除
    # assert B == batch_size and n == robot1_q_num + robot2_q_num

    # 保持你原来的 float64 计算逻辑，但避免多余拷贝
    J = jac_batch.to(torch.float64, copy=False)
    device = J.device

    # 单位阵用 expand（零拷贝），避免 repeat 的真实复制
    batch_i  = torch.eye(n, dtype=torch.float64, device=device).unsqueeze(0).expand(B, n, n)
    batch_eps = (1e-16) * torch.eye(m, dtype=torch.float32, device=device).unsqueeze(0).expand(B, m, m)

    # 按原逻辑计算
    J_t   = J.transpose(-2, -1).contiguous()  # 保证后续 matmul 的内存布局更友好
    J_J_t = torch.matmul(J, J_t)
    J_pinv = torch.matmul(J_t, torch.inverse(J_J_t + batch_eps))
    J_pinv_J = torch.matmul(J_pinv, J)
    rel_jacobian_null = batch_i - J_pinv_J
    return rel_jacobian_null.to(torch.float32)

##########################################################
################## CPU MODEL Define ######################
##########################################################

robot1_config_dh_mat = np.array([[0.0, 0.333,   0.0,        0.0, 0],
                                [0.0, 0.0,     0.0,    -1.5708, 0],
                                [0.0, 0.316,   0.0,     1.5708, 0],
                                [0.0, 0.0,     0.0825,  1.5708, 0],
                                [0.0, 0.384,  -0.0825, -1.5708, 0],
                                [0.0, 0.0,     0.0,     1.5708, 0],
                                [0.0, 0.0,   0.088,   1.5708, 0]])
robot1_dh_mat =  robot1_config_dh_mat.T

cpu_robot1 = DQ_SerialManipulatorMDH(robot1_dh_mat)
cpu_robot1.set_base_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, -0.175, 0.0]))
cpu_robot1.set_effector(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0]))
cpu_robot1.set_reference_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, -0.175, 0.0]))
# cpu robot2
robot2_config_dh_mat = np.array([[0.0, 0.333,   0.0,        0.0, 0],
                                [0.0, 0.0,     0.0,    -1.5708, 0],
                                [0.0, 0.316,   0.0,     1.5708, 0],
                                [0.0, 0.0,     0.0825,  1.5708, 0],
                                [0.0, 0.384,  -0.0825, -1.5708, 0],
                                [0.0, 0.0,     0.0,     1.5708, 0],
                                [0.0, 0.0,   0.088,   1.5708, 0]])
robot2_dh_mat =  robot2_config_dh_mat.T
cpu_robot2 = DQ_SerialManipulatorMDH(robot2_dh_mat)
cpu_robot2.set_base_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.175, 0.0]))
cpu_robot2.set_effector(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0]))
cpu_robot2.set_reference_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.175, 0.0]))
cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(cpu_robot1, cpu_robot2)
a = cpu_dq_dual_arm_model.relative_pose([0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853, 0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853])
q = [0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853, 0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853]

########################################################### 
################### GPU MODEL Define #####################
##########################################################    
batch_size = 1000
dh_matrix1 = torch.tensor([
    [0.0, 0.333,   0.0,        0.0, 0],
    [0.0, 0.0,     0.0,    -1.5708, 0],
    [0.0, 0.316,   0.0,     1.5708, 0],
    [0.0, 0.0,     0.0825,  1.5708, 0],
    [0.0, 0.384,  -0.0825, -1.5708, 0],
    [0.0, 0.0,     0.0,     1.5708, 0],
    [0.0, 0.0,   0.088,   1.5708, 0]
], dtype=torch.float32, device= "cuda:0")
dh_matrix2 = torch.tensor([
    [0.0, 0.333,   0.0,        0.0, 0],
    [0.0, 0.0,     0.0,    -1.5708, 0],
    [0.0, 0.316,   0.0,     1.5708, 0],
    [0.0, 0.0,     0.0825,  1.5708, 0],
    [0.0, 0.384,  -0.0825, -1.5708, 0],
    [0.0, 0.0,     0.0,     1.5708, 0],
    [0.0, 0.0,   0.088,   1.5708, 0]
], dtype=torch.float32, device= "cuda:0") 
dual_arm_joint_pos = [0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853, 0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853]
batch_robot1_base = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, -0.175, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot2_base = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, 0.175, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot1_effector = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot2_effector = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
q_vec1 = torch.tensor([0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1) 
q_vec2 = torch.tensor([0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
q_vec3 = torch.tensor([0, -1, 0, -2, -1.7, 1.4707, 0.7853], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
dh_matrix1 = dh_matrix1.reshape(-1)
dh_matrix2 = dh_matrix2.reshape(-1)
desire_line_d = [0,0,0,1]
desire_quat_line_ref = [0, -0.011682, 0.003006, -0.999927]
batch_line_d = torch.tensor(desire_line_d, dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_quat_line_ref = torch.tensor(desire_quat_line_ref, dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
first_bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian,  batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
first_bacth_rel_pos = torch.tensor([9.63267947e-05,  7.07173586e-01, -7.07039957e-01, -9.63267808e-05, 3.51225857e-06, 2.47457994e-01, 2.47504763e-01, -9.62904111e-07], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_ident_dq = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
temp_dq = dq_mult(conj(first_bacth_rel_pos), batch_ident_dq)
error_dq = batch_ident_dq - temp_dq 
error_dq = error_dq.reshape(batch_size, 8, 1)

bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian,  batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
for i in range(10):
    q1,q2 = project_q_by_constraint_jacobian(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec3, first_bacth_rel_pos,
                        7, 7, 1, 1)
bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian,  batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q1, q2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)

########################################################### 
##################### time test ###########################
##########################################################  
torch.cuda.synchronize()
start_time = time.time()
q1,q2 = project_q_by_constraint_jacobian(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec3, first_bacth_rel_pos,
                         7, 7, 1, 1)
torch.cuda.synchronize()
end_time = time.time()
print("Time taken for project_q_by_constraint_jacobian: ", end_time - start_time)

for i in range(10):
    get_rel_jacobian_null(bacth_rel_jacobian)

torch.cuda.synchronize()
start_time = time.time()
a = get_rel_jacobian_null(bacth_rel_jacobian)
torch.cuda.synchronize()
end_time = time.time()
print("Time taken for get_rel_jacobian_null: ", end_time - start_time)