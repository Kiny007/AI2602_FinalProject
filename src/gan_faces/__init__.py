"""GAN 人头图像生成项目源码包。"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Discriminator",
    "Generator",
    "StyleGeneratorLite",
    "init_dcgan_weights",
    "init_stylegan_lite_weights",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".models", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
