import os
import platform
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # トップレベルのbuildディレクトリを使用
        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        extdir = os.path.join(os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))), "rgf")
        pybind11_dir = os.path.join(
            sys.prefix,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
            "pybind11",
            "share",
            "cmake",
            "pybind11",
        )
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-Dpybind11_DIR={pybind11_dir}",
            "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
            "-DBUILD_PYTHON_BINDING=ON",
        ]

        if platform.system() == "Windows":
            cmake_args += ["-G", "Ninja"]

        build_temp = os.path.join(build_dir, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_temp)


setup(
    package_data={"rgf": ["*.so", "*.pyd"]},
    ext_modules=[CMakeExtension("rgf")],
    cmdclass=dict(build_ext=CMakeBuild),
)
