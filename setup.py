from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

CUDA = os.environ.get("TIDEGBM_CUDA", "").strip() not in {"", "0", "false", "False"}

sources = [
    "cpp/src/histogram.cpp",
    "cpp/src/binning.cpp",
    "cpp/src/tree.cpp",
    "cpp/src/objective.cpp",
    "cpp/src/booster.cpp",
    "cpp/src/serialize.cpp",
    "cpp/src/c_api.cpp",
    "cpp/src/bindings.cpp",
]

if sys.platform == "win32":
    compile_args = ["/O2", "/std:c++17", "/EHsc", "/openmp"]
    link_args = []
elif sys.platform == "darwin":
    compile_args = ["-O3", "-std=c++17", "-fvisibility=hidden", "-fno-math-errno"]
    link_args = []
    libomp = os.environ.get("TIDEGBM_LIBOMP_PREFIX")
    if not libomp and shutil.which("brew"):
        try:
            libomp = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            libomp = None
    if not libomp:
        raise RuntimeError(
            "macOS build requires libomp. Run: brew install libomp "
            "(cibuildwheel sets this via CIBW_BEFORE_ALL_MACOS)."
        )
    compile_args += ["-Xpreprocessor", "-fopenmp", f"-I{libomp}/include"]
    link_args += [
        f"-L{libomp}/lib",
        "-lomp",
        "-Wl,-rpath,@loader_path/.dylibs",
    ]
else:
    compile_args = ["-O3", "-std=c++17", "-fvisibility=hidden", "-ftree-vectorize", "-fno-math-errno"]
    link_args = []
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]
if CUDA:
    compile_args += ["-DTIDEGBM_CUDA"]
    sources.append("cpp/src/histogram_cuda.cu")


def cuda_home() -> str:
    env = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env:
        return env
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    if os.path.isdir("/usr/local/cuda"):
        return "/usr/local/cuda"
    raise RuntimeError(
        "TIDEGBM_CUDA is set but nvcc was not found. Install the CUDA toolkit and set CUDA_HOME."
    )


class build_ext(_build_ext):
    def build_extension(self, ext):
        cu = [s for s in ext.sources if s.endswith(".cu")]
        ext.sources = [s for s in ext.sources if not s.endswith(".cu")]
        if cu:
            home = cuda_home()
            nvcc = os.path.join(home, "bin", "nvcc")
            if not os.path.isfile(nvcc):
                found = shutil.which("nvcc")
                if not found:
                    raise RuntimeError(f"nvcc not found under {home}")
                nvcc = found
            os.makedirs(self.build_temp, exist_ok=True)
            objects = []
            gencodes = [
                "-gencode=arch=compute_60,code=sm_60",
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_90,code=sm_90",
            ]
            for src in cu:
                obj = os.path.join(self.build_temp, os.path.basename(src) + ".o")
                cmd = [
                    nvcc,
                    "-O3",
                    "-std=c++17",
                    "-DTIDEGBM_CUDA",
                    "-Xcompiler",
                    "-fPIC,-fvisibility=hidden",
                    "--cudart",
                    "static",
                    "-I",
                    "cpp/include",
                    "-c",
                    src,
                    "-o",
                    obj,
                    *gencodes,
                ]
                print("nvcc:", " ".join(cmd), file=sys.stderr)
                subprocess.check_call(cmd)
                objects.append(obj)
            ext.extra_objects = list(ext.extra_objects or []) + objects
            lib64 = os.path.join(home, "lib64")
            if os.path.isdir(lib64):
                ext.library_dirs = list(ext.library_dirs or []) + [lib64]
            ext.extra_link_args = list(ext.extra_link_args or []) + [
                "-lcudart_static",
                "-ldl",
                "-lrt",
                "-lpthread",
            ]
        super().build_extension(ext)


ext = Extension(
    "tidegbm._core",
    sources=sources,
    include_dirs=["cpp/include", pybind11.get_include()],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="tidegbm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext} if CUDA else {},
    zip_safe=False,
)
