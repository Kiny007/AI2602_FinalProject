from pathlib import Path
from typing import Optional

from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceImageFolder(Dataset):
    """读取普通图片文件夹，允许图片分散在任意子目录中。"""

    def __init__(self, root: str | Path, transform: Optional[transforms.Compose] = None) -> None:
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
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


class ImageOnlyDataset(Dataset):
    """包装 torchvision 数据集，只保留图片，丢弃标签或属性。"""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        # torchvision 的人脸数据集通常返回 (image, target)，训练 GAN 只需要 image。
        if isinstance(item, tuple):
            return item[0]
        return item


def build_face_transform(image_size: int) -> transforms.Compose:
    """统一图片尺寸并归一化到 [-1, 1]，匹配生成器最后的 Tanh 输出。"""

    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_dataset(
    dataset_name: str,
    data_root: str | Path,
    image_size: int,
    download: bool = False,
) -> Dataset:
    """根据命令行参数构建数据集。"""

    name = dataset_name.lower()
    transform = build_face_transform(image_size)
    data_root = Path(data_root)

    if name == "folder":
        return FaceImageFolder(data_root, transform=transform)

    if name == "lfw":
        # LFW 数据集较小，适合作为课程项目的快速实验数据。
        dataset = datasets.LFWPeople(
            root=str(data_root),
            split="train",
            image_set="funneled",
            transform=transform,
            download=download,
        )
        return ImageOnlyDataset(dataset)

    if name == "celeba":
        # CelebA 更适合生成任务，但通常需要用户提前准备数据。
        dataset = datasets.CelebA(
            root=str(data_root),
            split="train",
            target_type="attr",
            transform=transform,
            download=download,
        )
        return ImageOnlyDataset(dataset)

    raise ValueError(f"不支持的数据集类型: {dataset_name}")


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """创建 DataLoader，pin_memory 在有 GPU 时可加速数据搬运。"""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
