"""Build script for native BitNet GEMV kernel."""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Detect CPU features
def get_compile_args():
    args = ['-O3', '-ffast-math', '-fopenmp', '-march=native']

    # Check for AVX512 support
    try:
        import subprocess
        result = subprocess.run(['grep', '-o', 'avx512', '/proc/cpuinfo'],
                              capture_output=True, text=True)
        if 'avx512' in result.stdout:
            args.extend(['-mavx512f', '-mavx512bw', '-mavx512vl'])
            print("AVX512 support detected")
    except:
        pass

    # Always include AVX2 as fallback
    args.append('-mavx2')

    return args

setup(
    name='bitnet_native',
    ext_modules=[
        CppExtension(
            'bitnet_native',
            ['bitnet_kernel.cpp'],
            extra_compile_args=get_compile_args(),
            extra_link_args=['-fopenmp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
