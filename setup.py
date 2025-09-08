import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, 'include')]
sources = glob.glob('*.cu') + glob.glob('*.cpp')

setup(
    name='dq_torch',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='dq_torch',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-DNDEBUG", "-std=c++17"],
                "nvcc": [
                    "-O3","-DNDEBUG","-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lineinfo","-Xptxas","-O3",
                    "-use_fast_math","-Xptxas","-dlcm=ca",
                ],
                }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },

    package_data={
        '': ['*.pyi'], 
    },

    packages=[''],
)
