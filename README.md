# cuda_dq_kernel
## Status

⚠️ **Experimental & unstable.** There are known issues with computation accuracy, and overall compute speed requires further improvement. Expect breaking API changes.

## Introduction
This library is an optimized adaptation built upon dq robotics, featuring a CUDA-based rewrite to enable accelerated batch computations.
| Project | Link|
| --------------------------| ------------------------------------------------------------------------------------- |
| dq robotics | https://github.com/dqrobotics/cpp|
## Env
- Ubuntu22.04
- CUDA 11.8
- Pytoch 2.0.1
## Install
```shell
pip insatll .
``` 
## Performance
batch size = 20000 tested on a  NVIDIA GeForce RTX™ 4080 laptops GPU. The relevant test codes can be found in: test_dual_franka_time.py and test_dual_ur_time.py.

Task is calculating the time for performing 20,000 iterations of `relative_pose`、`absolute_pose`、`relative_pose_jacobian`、`absolute_pose_jacobian`.

|| Intel Core i9-14900HX |  NVIDIA GeForce RTX™ 4080 laptop GPU  |
| --- | -----| -----|
| UR3 and UR3e | 260 ms | 1.1 ms |
| Two Franka | 330 ms | 1.6 ms |

## What’s Parallelized

The implementation executes **N samples/instances per launch** for high throughput. Current batch‑parallel CUDA entry points include:

### Composite kernels (pose + Jacobian)

* `rel_abs_pose_rel_jac_cuda`

  * Batched relative pose ,absolute pose relative Jacobian.
* `rel_abs_pose_rel_abs_jac_cuda`

  * Batched relative & absolute poses and their Jacobians.

### Primitive dual‑quaternion & geometry ops

* `dq_mult_cuda`, `dq_exp_cuda`, `dq_log_cuda`, `dq_sqrt_cuda`, `dq_inv_cuda`, `dq_normalize_cuda`
* `P_cuda`, `D_cuda`, `Re_cuda`, `Im_cuda`, `conj_cuda`, `hamiplus8_cuda`, `haminus8_cuda`
* `norm_cuda`, `rotation_axis_cuda`, `rotation_angle_cuda`, `translation_cuda`

> These are implemented as CUDA kernels and wrapped for PyTorch. All listed ops support **batched inputs**.







