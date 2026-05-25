# 基于 DCGAN 的人头图像生成

本项目根据 `project.pdf` 的基本要求搭建：使用 DCGAN 训练人头图像生成器，支持生成样例、潜变量线性插值，并使用 Inception Score（IS）评估生成质量。

## 项目结构

```text
.
├── train.py                 # 训练 DCGAN
├── generate.py              # 使用训练好的生成器生成图片
├── interpolate.py           # 在两张生成头像之间做线性插值
├── evaluate.py              # 使用 IS 评估生成图像质量
├── src/gan_faces/
│   ├── data.py              # 数据集读取和预处理
│   ├── models.py            # DCGAN 生成器与判别器
│   ├── metrics.py           # Inception Score 实现
│   └── utils.py             # 随机种子、保存图片、加载模型等工具
└── docs/project_requirements.md
```

## 环境安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 数据准备

推荐两种方式：

1. 使用本地人脸图片文件夹，图片可以直接放在 `data/faces/` 或任意子目录中。
2. 使用 torchvision 的 LFW 数据集，小规模实验更轻量：

```powershell
python train.py --dataset lfw --data-root data --download --epochs 20
```

CelebA 体积较大，且下载通常需要手动授权；如果已经准备好数据，可以使用：

```powershell
python train.py --dataset celeba --data-root data --epochs 50
```

## 训练 DCGAN

本地图片文件夹示例：

```powershell
python train.py --dataset folder --data-root data/faces --epochs 50 --batch-size 128
```

训练输出默认保存在 `outputs/`：

- `outputs/checkpoints/latest.pt`：最新模型权重
- `outputs/checkpoints/dcgan_epoch_XXXX.pt`：阶段性模型权重
- `outputs/samples/epoch_XXXX.png`：固定噪声生成的训练过程样例
- `outputs/train_log.csv`：训练损失日志

## 生成头像

```powershell
python generate.py --checkpoint outputs/checkpoints/latest.pt --num-images 64 --output outputs/generated/grid.png
```

## 两张头像之间的线性插值

DCGAN 没有编码器，因此这里按照 GAN 常见做法：选取两个潜变量作为两张生成头像的端点，在潜变量空间中做线性插值，并观察生成头像的连续变化。

```powershell
python interpolate.py --checkpoint outputs/checkpoints/latest.pt --steps 12 --output outputs/interpolation/linear.png
```

## IS 评估

```powershell
python evaluate.py --checkpoint outputs/checkpoints/latest.pt --num-images 5000 --batch-size 64
```

首次运行 IS 评估时，torchvision 可能会下载 Inception v3 的预训练权重。输出格式类似：

```text
Inception Score: mean=2.3142, std=0.0821
```

## 已覆盖的基本要求

| 基本要求 | 对应实现 |
| --- | --- |
| 实现基础 GAN 模型（DCGAN） | `src/gan_faces/models.py` |
| 在数据集上训练并生成头像 | `train.py`、`generate.py` |
| 测试两张头像之间线性插值 | `interpolate.py` |
| 使用 FID 或 IS 评估质量 | `evaluate.py` 使用 IS |
