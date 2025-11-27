from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        # Ensure .mm is recognized
        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")
        for ext in self.extensions:
            ext.extra_compile_args = ext.extra_compile_args or []
            if self.compiler.compiler_type == "unix":
                ext.extra_compile_args += ["-std=c++17", "-ObjC++"]
        super().build_extensions()


ext = Extension(
    "nax_check",
    sources=["nax_check.mm"],
    extra_link_args=["-framework", "Metal"],
    language="objc++",
)

setup(
    name="nax_check",
    version="0.0.1",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
)
