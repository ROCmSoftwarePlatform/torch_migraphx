#####################################################################################
# Copyright (c) 2022-present, Advanced Micro Devices, Inc. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#####################################################################################

import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand

__version__ = open("version.txt").read().strip()

TORCH_CMAKE_PATH = os.environ.get("TORCH_CMAKE_PATH")
if not TORCH_CMAKE_PATH:
    import torch
    TORCH_CMAKE_PATH = torch.utils.cmake_prefix_path

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_PREFIX_PATH=' + TORCH_CMAKE_PATH
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            # TODO: find # of procs
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp,
                              env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)

        print()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='torch_migraphx',
    version=__version__,
    author='AMD',
    author_email='Shivad.Bhavsar@amd.com',
    url='https://github.com/ROCmSoftwarePlatform/torch_migraphx',
    description='Intergrate PyTorch with MIGraphX acceleration engine',
    long_description_content_type='text/markdown',
    long_description=long_description,
    setup_requires=["cmake"],
    install_requires=[
    "torch>=1.11.0",
    "numpy>=1.20.0",
    "pybind11-global",
    "packaging",
    "tabulate",
    ],
    packages=find_packages(),
    package_dir={'torch_migraphx': 'torch_migraphx'},
    ext_modules=[CMakeExtension('_torch_migraphx')],
    cmdclass=dict(build_ext=CMakeBuild),
    license="BSD",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
