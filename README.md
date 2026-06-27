# 基于 GAN 的人头图像生成

本项目使用 DCGAN 完成基础人头图像生成，支持多卡混合精度训练、生成样例、潜变量线性插值、和 ID/FID 等多指标评估，对比了 BCE Loss、Wasserstein Loss、Hinge Loss 等损失函数和 梯度惩罚、谱归一化等正则化归一化方法，最后复现了 StyleGAN 模型，比较其与基础 DCGAN 的性能差异。

<!-- ## 项目结构

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
``` -->

## 环境安装

```powershell
conda create -n gan python==3.10
pip install -r requirements.txt
```

## 数据准备

推荐两种方式：

1. 使用本地人脸图片文件夹，图片可以直接放在 `data/faces/` 或任意子目录中。
2. 使用 torchvision 的 LFW 数据集，小规模实验更轻量：

```powershell
python train.py --dataset lfw --data-root data --download --epochs 20
```



> [!NOTE]
> 为了减少训练过程的cpu操作，我们把对图像的裁剪、统一格式等预处理放到了``pack_celeba.py``中，参考了英伟达StyleGAN2的实现，在训练前，先运行以下代码
> ```
> python pack_celeba.py --source data/celeba/img_align_celeba --dest data/celeba_64.zip --size 64
> ```

## 训练 DCGAN

本地图片文件夹示例：

```powershell
python train.py --dataset folder --data-root data/celeba/img_align_celeba --output-dir outputs/dcgan --epochs 50 --batch-size 64
```

> [!NOTE]
> 本仓库使用accelerate管理多卡进程和混合精度等，先通过下面的命令设置卡数
> ```
> accelerate config
> ```
>
> 为了更直观地设置超参，可以先在``src/gan_faces/config/``中配置超参数的yaml文件，启动训练时直接传入这个yaml文件即可。
> 其中，如果数据集传的是前面准备好的一个zip文件
> ```
> dataset: folder
> data_root: data/celeba_64.zip
> ```
> 如果要从断点恢复训练，再通过``--resume``传入ckpt的地址
> ```
> accelerate launch train.py --config src/gan_faces/config/dcgan.yaml --resume .\outputs\dcgan_acc\checkpoints\latest.pt
> ```

训练输出默认保存在 `outputs/`，建议对比实验时显式指定 `outputs/dcgan`：

- `outputs/dcgan/checkpoints/latest.pt`：最新模型权重
- `outputs/dcgan/checkpoints/dcgan_epoch_XXXX.pt`：阶段性模型权重
- `outputs/dcgan/samples/epoch_XXXX.png`：固定噪声生成的训练过程样例
- `outputs/dcgan/train_log.csv`：训练损失日志
- `outputs/dcgan/tensorboard/`：TensorBoard 事件文件






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
## 曲线可视化
为了更清晰地展现训练过程中各指标的变化，我们设计了三个可视化脚本
- ``plot_d_real_fake_curves.py`` 可视化了 BCE Loss 下判别器对合成图片和真实图片的评分
- ``plot_loss_curves.py`` 可视化了生成器和判别器的损失曲线
- ``plot_metric_cureve.py`` 可视化了FID变化情况 

请从tensorboard中下载对应曲线的csv文件，放到对应文件夹，然后运行
```
python plot_d_real_fake_curves.py --input-dir outputs/[path to your folder of csvs]
python plot_loss_curves.py --input-dir outputs/[path to your folder of csvs] 
python plot_metric_curve.py --input-csv outputs/[path to your fid csv]  --title "FID5K Curve" --ylabel "FID5K" --color "#9467bd" 
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


## Bonus：训练 StyleGAN

为了对比改进 GAN 模型与基础模型性能差异，本项目基于英伟达官方仓库实现了 StyleGAN2 和 StyleGAN-ada，具体操作参见[此代码仓库](https://github.com/sunnyxrxrx/stylegan2-ada-pytorch)。

本仓库已经适配 StyleGAN 模型的评估。

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
| 对比改进 GAN 模型与基础模型性能差异 | 参见[此代码仓库](https://github.com/sunnyxrxrx/stylegan2-ada-pytorch) |
