"""数据集和 DataLoader 构建工具。

本文件负责把本地图片文件夹、torchvision 人脸数据集以及 CycleGAN 的
无配对 A/B 双域图片统一转换为训练脚本可直接使用的 PyTorch Dataset。
"""

import random
import io
import json
import zipfile
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceImageFolder(Dataset):
    """读取普通图片文件夹，允许图片分散在任意子目录中。"""

    def __init__(self, root: Union[str, Path], transform: Optional[transforms.Compose] = None) -> None:
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.root}")

        # 递归扫描所有常见图片格式，方便直接使用自建人脸数据集。
        self.paths = sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"未在 {self.root} 中找到图片文件")

    def __len__(self) -> int:
        """返回可用图片数量。"""

        return len(self.paths)

    def __getitem__(self, index: int):
        """读取单张图片，转换为 RGB 后应用预处理。"""

        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class PackedImageZip(Dataset):
    """读取预处理好的 zip 图片包，避免训练时重复做尺寸变换。"""

    def __init__(self, path: Union[str, Path], transform: Optional[transforms.Compose] = None) -> None:
        self.path = Path(path)
        self.transform = transform
        self._zipfile: zipfile.ZipFile | None = None
        self.metadata: dict = {}

        if not self.path.exists():
            raise FileNotFoundError(f"数据压缩包不存在: {self.path}")
        if self.path.suffix.lower() != ".zip":
            raise ValueError(f"PackedImageZip 仅支持 .zip 文件: {self.path}")

        with zipfile.ZipFile(self.path, "r") as zf:
            all_names = zf.namelist()
            self.paths = sorted(
                name
                for name in all_names
                if Path(name).suffix.lower() in IMG_EXTENSIONS and not name.endswith("/")
            )
            if not self.paths:
                raise ValueError(f"未在 {self.path} 中找到图片文件")
            if "dataset.json" in all_names:
                with zf.open("dataset.json", "r") as file:
                    self.metadata = json.load(file)

    def _get_zipfile(self) -> zipfile.ZipFile:
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.path, "r")
        return self._zipfile

    def close(self) -> None:
        if self._zipfile is not None:
            self._zipfile.close()
            self._zipfile = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zipfile"] = None
        return state

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with self._get_zipfile().open(self.paths[index], "r") as file:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class UnpairedImageFolder(Dataset):
    """读取 CycleGAN 所需的两个无配对图片域。"""

    def __init__(
        self,
        root_a: Union[str, Path],
        root_b: Union[str, Path],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.domain_a = FaceImageFolder(root_a, transform=transform)
        self.domain_b = FaceImageFolder(root_b, transform=transform)

    def __len__(self) -> int:
        """返回较大域的长度，让较小域可以重复随机采样。"""

        return max(len(self.domain_a), len(self.domain_b))

    def __getitem__(self, index: int):
        """返回一张 A 域图片和一张随机 B 域图片，二者不要求成对。"""

        image_a = self.domain_a[index % len(self.domain_a)]
        image_b = self.domain_b[random.randrange(len(self.domain_b))]
        return image_a, image_b


class ImageOnlyDataset(Dataset):
    """包装 torchvision 数据集，只保留图片，丢弃标签或属性。"""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        """返回被包装 torchvision 数据集的样本数量。"""

        return len(self.dataset)

    def __getitem__(self, index: int):
        """丢弃标签或属性，只返回图片张量。"""

        item = self.dataset[index]
        # torchvision 的人脸数据集通常返回 (image, target)，训练 GAN 只需要 image。
        if isinstance(item, tuple):
            return item[0]
        return item


def build_face_transform(image_size: int, presized: bool = False) -> transforms.Compose:
    """统一图片尺寸并归一化到 [-1, 1]，匹配生成器最后的 Tanh 输出。"""

    transform_steps = []
    if not presized:
        transform_steps.extend(
            [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(transform_steps)


def build_dataset(
    dataset_name: str,
    data_root: Union[str, Path],
    image_size: int,
    download: bool = False,
) -> Dataset:
    """根据命令行参数构建数据集。"""

    name = dataset_name.lower()
    data_root = Path(data_root)

    if name == "folder":
        if data_root.suffix.lower() == ".zip":
            dataset = PackedImageZip(data_root, transform=build_face_transform(image_size, presized=True))
            packed_resolution = dataset.metadata.get("resolution")
            if packed_resolution is not None and int(packed_resolution) != image_size:
                raise ValueError(
                    f"压缩包分辨率为 {packed_resolution}，但当前请求 image_size={image_size}"
                )
            return dataset
        return FaceImageFolder(data_root, transform=build_face_transform(image_size))

    if name == "lfw":
        # LFW 数据集较小，适合作为课程项目的快速实验数据。
        dataset = datasets.LFWPeople(
            root=str(data_root),
            split="train",
            image_set="funneled",
            transform=build_face_transform(image_size),
            download=download,
        )
        return ImageOnlyDataset(dataset)

    if name == "celeba":
        # CelebA 更适合生成任务，但通常需要用户提前准备数据。
        dataset = datasets.CelebA(
            root=str(data_root),
            split="train",
            target_type="attr",
            transform=build_face_transform(image_size),
            download=download,
        )
        return ImageOnlyDataset(dataset)

    raise ValueError(f"不支持的数据集类型: {dataset_name}")


def build_unpaired_dataset(
    domain_a_root: Union[str, Path],
    domain_b_root: Union[str, Path],
    image_size: int,
) -> Dataset:
    """为 CycleGAN 构建 A/B 两个无配对图像域。"""

    transform = build_face_transform(image_size)
    return UnpairedImageFolder(domain_a_root, domain_b_root, transform=transform)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = True,
    sampler: Sampler | None = None,
) -> DataLoader:
    """创建 DataLoader，pin_memory 在有 GPU 时可加速数据搬运。"""

    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
