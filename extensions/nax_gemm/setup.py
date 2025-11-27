import inspect
import os
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import numpy


def find_mlx_include():
    try:
        import mlx

        import mlx.core  # noqa
        mlx_dir = Path(inspect.getfile(mlx)).parent
        inc = mlx_dir / "include"
        if inc.exists():
            return str(inc)
    except Exception:
        pass
    return None


class BuildExt(build_ext):
    def build_extensions(self):
        # Make sure .mm is recognized
        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")

        # Build Metal shader -> metallib
        mlx_inc = find_mlx_include()
        metal_src = Path(__file__).parent / "nax_gemm.metal"
        air = Path(self.build_temp) / "nax_gemm.air"
        metallib = Path(self.build_temp) / "nax_gemm.metallib"
        metal_cmd = [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(metal_src),
            "-o",
            str(air),
        ]
        if mlx_inc:
            metal_cmd.extend(["-I", mlx_inc])
        metallib_cmd = [
            "xcrun",
            "-sdk",
            "macosx",
            "metallib",
            str(air),
            "-o",
            str(metallib),
        ]
        try:
            os.makedirs(self.build_temp, exist_ok=True)
            subprocess.check_call(metal_cmd)
            subprocess.check_call(metallib_cmd)
            # Copy metallib next to extension output
            for ext in self.extensions:
                outdir = Path(self.get_ext_fullpath(ext.name)).parent
                outdir.mkdir(parents=True, exist_ok=True)
                self.copy_file(str(metallib), outdir / "nax_gemm.metallib")
        except Exception as e:
            print("Warning: Metal build failed:", e)

        # Add ObjC++ flags
        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args or []
            if self.compiler.compiler_type == "unix":
                ext.extra_compile_args += ["-std=c++17", "-ObjC++"]
            ext.extra_link_args = ext.extra_link_args or []
            ext.extra_link_args += ["-framework", "Metal"]

        super().build_extensions()


ext = Extension(
    "nax_gemm",
    sources=["nax_gemm.mm"],
    language="objc++",
    include_dirs=[numpy.get_include()],
)

setup(
    name="nax_gemm",
    version="0.0.1",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
)
