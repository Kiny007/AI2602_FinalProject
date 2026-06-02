# 基于 GAN 的人头图像生成

本项目根据 `project.pdf` 搭建：使用 DCGAN 完成基础人头图像生成，支持生成样例、潜变量线性插值、TensorBoard 训练可视化、Inception Score（IS）和 FID 评估，并补充 Bonus 1：对比轻量 StyleGAN 风格模型或 CycleGAN 与基础 DCGAN 的性能差异。

## 项目结构

```text
.
├── train.py                 # 训练 DCGAN
├── train_stylegan.py        # 训练轻量 StyleGAN 风格模型
├── train_cyclegan.py        # 训练 CycleGAN 无配对图像域转换模型
├── generate.py              # 使用训练好的生成器生成图片
├── interpolate.py           # 在两张生成头像之间做线性插值
├── evaluate.py              # 使用 IS/FID 评估生成图像质量
├── compare_models.py        # 对比 DCGAN 与 StyleGAN-Lite
├── compare_gan_cyclegan.py  # 对比 DCGAN 与 CycleGAN
├── src/gan_faces/
│   ├── data.py              # 数据集读取和预处理
│   ├── models.py            # DCGAN、StyleGAN-Lite、CycleGAN 与判别器
│   ├── metrics.py           # Inception Score 和 FID 实现
│   ├── tensorboard.py       # DCGAN TensorBoard 日志工具
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
python train.py --dataset folder --data-root data/celeba/img_align_celeba --output-dir outputs/dcgan --epochs 50 --batch-size 64
```

训练输出默认保存在 `outputs/`，建议对比实验时显式指定 `outputs/dcgan`：

- `outputs/dcgan/checkpoints/latest.pt`：最新模型权重
- `outputs/dcgan/checkpoints/dcgan_epoch_XXXX.pt`：阶段性模型权重
- `outputs/dcgan/samples/epoch_XXXX.png`：固定噪声生成的训练过程样例
- `outputs/dcgan/train_log.csv`：训练损失日志
- `outputs/dcgan/tensorboard/`：TensorBoard 事件文件

查看 TensorBoard：

```powershell
tensorboard --logdir outputs/dcgan/tensorboard
```

如果只想保存 CSV 和图片、不写 TensorBoard：

```powershell
python train.py --dataset folder --data-root data/celeba/img_align_celeba --output-dir outputs/dcgan --no-tensorboard
```

## Bonus 1：训练 StyleGAN-Lite

为了完成“对比改进 GAN 模型与基础模型性能差异”，本项目实现了一个适合 64x64 CelebA 实验的轻量 StyleGAN 风格生成器。它包含 mapping network、learned constant、AdaIN 和 noise injection，但没有实现完整 StyleGAN/StyleGAN2 的全部训练技巧。

```powershell
python train_stylegan.py --dataset folder --data-root data/celeba/img_align_celeba --output-dir outputs/stylegan_lite --epochs 50 --batch-size 64
```

训练后会得到：

- `outputs/stylegan_lite/checkpoints/latest.pt`
- `outputs/stylegan_lite/samples/epoch_XXXX.png`
- `outputs/stylegan_lite/train_log.csv`

## Bonus 1：训练 CycleGAN

如果选择用 CycleGAN 完成“改进 GAN 与基础 GAN 的性能差异对比”，需要准备两个无配对图片域。例如：

- 域 A：普通人脸头像 `data/domain_a/`
- 域 B：目标风格头像 `data/domain_b/`

训练命令：

```powershell
python train_cyclegan.py --domain-a-root data/domain_a --domain-b-root data/domain_b --output-dir outputs/cyclegan --epochs 50 --batch-size 4
```

训练后会得到：

- `outputs/cyclegan/checkpoints/latest.pt`
- `outputs/cyclegan/samples/epoch_XXXX.png`
- `outputs/cyclegan/train_log.csv`

CycleGAN 是图像到图像翻译模型，不是从随机噪声直接生成头像；因此对比时脚本会用真实 A 域图片作为输入，评估 A->B 翻译结果。

## 生成头像

```powershell
python generate.py --checkpoint outputs/dcgan/checkpoints/latest.pt --num-images 64 --output outputs/generated/dcgan_grid.png
python generate.py --checkpoint outputs/stylegan_lite/checkpoints/latest.pt --num-images 64 --output outputs/generated/stylegan_lite_grid.png
```

## 两张头像之间的线性插值

DCGAN 没有编码器，因此这里按照 GAN 常见做法：选取两个潜变量作为两张生成头像的端点，在潜变量空间中做线性插值，并观察生成头像的连续变化。

```powershell
python interpolate.py --checkpoint outputs/dcgan/checkpoints/latest.pt --steps 12 --output outputs/interpolation/dcgan_linear.png
python interpolate.py --checkpoint outputs/stylegan_lite/checkpoints/latest.pt --steps 12 --output outputs/interpolation/stylegan_lite_linear.png
```

## IS / FID 评估

```powershell
python evaluate.py --metric is --checkpoint outputs/dcgan/checkpoints/latest.pt --num-images 5000 --batch-size 64 --output-json outputs/metrics/dcgan_is.json
python evaluate.py --metric fid --checkpoint outputs/dcgan/checkpoints/latest.pt --dataset folder --data-root data/celeba/img_align_celeba --num-images 5000 --batch-size 64 --output-json outputs/metrics/dcgan_fid.json
python evaluate.py --metric both --checkpoint outputs/dcgan/checkpoints/latest.pt --dataset folder --data-root data/celeba/img_align_celeba --num-images 5000 --batch-size 64 --output-json outputs/metrics/dcgan_is_fid.json
```

首次运行 IS/FID 评估时，torchvision 可能会下载 Inception v3 的预训练权重。FID 会把生成图片与真实数据集图片的 Inception 特征分布做比较，数值越低通常越好。输出格式类似：

```text
Inception Score: mean=2.3142, std=0.0821
FID: 85.3721
```

## 模型性能对比

训练完两个模型后运行：

```powershell
python compare_models.py --dcgan-checkpoint outputs/dcgan/checkpoints/latest.pt --stylegan-checkpoint outputs/stylegan_lite/checkpoints/latest.pt --num-images 5000 --batch-size 64
```

脚本会对比：

- 生成器参数量
- 生成速度 images/s
- Inception Score 均值与标准差

结果保存到：

- `outputs/metrics/model_comparison.json`
- `outputs/metrics/model_comparison.csv`

## 基础 GAN 与 CycleGAN 性能对比

训练完 DCGAN 和 CycleGAN 后运行：

```powershell
python compare_gan_cyclegan.py --dcgan-checkpoint outputs/dcgan/checkpoints/latest.pt --cyclegan-checkpoint outputs/cyclegan/checkpoints/latest.pt --domain-a-root data/domain_a --domain-b-root data/domain_b --direction a2b --num-images 1000 --batch-size 64
```

脚本会对比：

- 生成器参数量和完整训练模型参数量
- 生成速度 images/s
- Inception Score 均值与标准差
- CycleGAN 的循环重建误差 `cycle_l1`

结果保存到：

- `outputs/metrics/gan_vs_cyclegan.json`
- `outputs/metrics/gan_vs_cyclegan.csv`

报告中需要说明：DCGAN 的输入是随机噪声，CycleGAN 的输入是真实源域图片；二者任务不同，IS 和速度可以横向参考，但 CycleGAN 的 `cycle_l1` 只用于判断自身的域转换一致性。

## 已覆盖的基本要求

| 基本要求 | 对应实现 |
| --- | --- |
| 实现基础 GAN 模型（DCGAN） | `src/gan_faces/models.py` |
| 在数据集上训练并生成头像 | `train.py`、`generate.py` |
| 测试两张头像之间线性插值 | `interpolate.py` |
| 使用 FID 或 IS 评估质量 | `evaluate.py` 支持 IS/FID |

## 已覆盖的 Bonus

| Bonus 任务 | 对应实现 |
| --- | --- |
| 对比改进 GAN 模型与基础模型性能差异 | `train_stylegan.py`、`compare_models.py`、`train_cyclegan.py`、`compare_gan_cyclegan.py`、`src/gan_faces/models.py` |
