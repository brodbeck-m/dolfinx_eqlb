# Copyright (C) 2024 Maximilian Brodbeck
#
# This file is part of dolfinx_eqlb
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import platform
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


VERSION = "1.1.0"

REQUIREMENTS = ["fenics-dolfinx==0.6.0", "numpy>=1.21.0"]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the"
                + "following extensions:"
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            raise RuntimeError("Windows not supported")
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j3"]

        env = os.environ.copy()
        import pybind11

        env["pybind11_DIR"] = pybind11.get_cmake_dir()
        env["CXXFLAGS"] = (
            f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp, env=env
        )


setup(
    name="dolfinx_eqlb",
    version=VERSION,
    description="dolfinX implementation of H(div) conforming flux reconstruction",
    author="Maximilian Brodbeck",
    author_email="maximilian.brodbeck@isd.uni-stuttgart.de",
    python_requires=">3.7.0",
    packages=[
        "dolfinx_eqlb",
        "dolfinx_eqlb.elmtlib",
        "dolfinx_eqlb.eqlb",
        "dolfinx_eqlb.lsolver",
    ],
    package_data={
        "dolfinx_eqlb.wrappers": ["*.h"],
        "dolfinx_eqlb": ["py.typed"],
        "dolfinx_eqlb.elmtlib": ["py.typed"],
        "dolfinx_eqlb.eqlb": ["py.typed"],
        "dolfinx_contact.lsolver": ["py.typed"],
    },
    ext_modules=[CMakeExtension("dolfinx_eqlb.cpp")],
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires=REQUIREMENTS,
    zip_safe=False,
)
