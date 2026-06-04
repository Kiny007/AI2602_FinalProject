"""将 CelebA 图片目录离线打包成固定分辨率 zip，减少训练期 CPU 预处理。"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 CelebA 打包成固定分辨率 zip")
    parser.add_argument("--source", type=str, required=True, help="原始图片目录")
    parser.add_argument("--dest", type=str, required=True, help="输出 zip 路径，例如 data/celeba_64.zip")
    parser.add_argument("--size", type=int, required=True, help="目标分辨率，例如 64")
    parser.add_argument(
        "--resize-filter",
        choices=["box", "lanczos"],
        default="lanczos",
        help="离线缩放时使用的滤波器",
    )
    parser.add_argument("--max-images", type=int, default=0, help="只打包前 N 张，<=0 表示全部")
    return parser.parse_args()


def center_crop_resize(image: Image.Image, size: int, resample: int) -> Image.Image:
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    return image.resize((size, size), resample=resample)


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    dest = Path(args.dest)
    if not source.exists():
        raise FileNotFoundError(f"源目录不存在: {source}")
    if dest.suffix.lower() != ".zip":
        raise ValueError(f"输出必须是 .zip 文件: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    resample = {"box": Image.BOX, "lanczos": Image.LANCZOS}[args.resize_filter]
    image_paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMG_EXTENSIONS)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"未在 {source} 中找到图片")

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as zf:
        labels = []
        for idx, image_path in enumerate(tqdm(image_paths, total=len(image_paths), desc="Packaging images")):
            with Image.open(image_path) as image:
                image = center_crop_resize(image.convert("RGB"), args.size, resample)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
            archive_name = f"{idx:08d}.png"
            zf.writestr(archive_name, buffer.getvalue())
            labels.append([archive_name, 0])

        metadata = {
            "name": dest.stem,
            "resolution": args.size,
            "num_images": len(image_paths),
            "labels": labels,
        }
        zf.writestr("dataset.json", json.dumps(metadata, indent=2))

    print(f"已打包 {len(image_paths)} 张图片 -> {dest}")


if __name__ == "__main__":
    main()
