from setuptools import Extension, setup  # type: ignore[import-untyped]
from setuptools.command.build_ext import build_ext  # type: ignore[import-untyped]

ext_modules = [
    Extension(
        "ringbuffer",
        sources=[
            "native/ringbuffer/ringbuffer.c",
            "native/ringbuffer/module.c",
        ],
        include_dirs=["native/ringbuffer"],
    )
]


class BuildExt(build_ext):  # type: ignore[misc]
    def build_extensions(self) -> None:
        # Add compiler flags for C99
        for ext in self.extensions:
            ext.extra_compile_args = ["-std=c99", "-Wall", "-Wextra", "-O2"]
        super().build_extensions()


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
