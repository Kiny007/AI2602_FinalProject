# project.pdf 要求摘要

任务：基于 GAN 的人头图像生成。

## 基本要求

1. 实现基础 GAN 模型，例如 DCGAN，并用于人头图像生成。
2. 在选定数据集上训练模型，生成较高质量的人头图像。
3. 测试两张人头图像之间线性插值的一系列结果。
4. 使用 FID 或 IS 评估生成图像质量。

## 当前项目对应实现

1. `src/gan_faces/models.py`：DCGAN 生成器、轻量 StyleGAN 风格生成器和判别器。
2. `train.py`：训练入口，支持本地图片文件夹、LFW、CelebA。
3. `generate.py`：加载训练权重并生成图片网格。
4. `interpolate.py`：在两个潜变量端点之间线性插值，生成连续头像变化图。
5. `evaluate.py` 和 `src/gan_faces/metrics.py`：使用 Inception Score 评估生成质量。
6. `train_stylegan.py`：训练轻量 StyleGAN 风格生成器，用于改进模型实验。
7. `compare_models.py`：比较 DCGAN 与 StyleGAN-Lite 的参数量、生成速度和 IS。

## Bonus 1 对应实现

Bonus 1 要求：对比改进的 GAN 模型（如 StyleGAN 或 CycleGAN）与基础模型的性能差异。

当前项目选择 StyleGAN 思路，新增 `StyleGeneratorLite`。该模型保留了 StyleGAN 的几个核心组件：

1. Mapping Network：将随机噪声 `z` 映射到风格空间 `w`。
2. Learned Constant：从可学习的 4x4 常量特征图开始生成。
3. AdaIN：用风格向量控制每层特征的通道缩放和平移。
4. Noise Injection：注入逐像素噪声，模拟发丝、皮肤纹理等局部随机细节。

对比实验建议使用相同数据集、相同训练轮数和相同 IS 评估图片数量：

```powershell
python train.py --dataset folder --data-root data/celeba/img_align_celeba --output-dir outputs/dcgan --epochs 50 --batch-size 64
python train_stylegan.py --dataset folder --data-root data/celeba/img_align_celeba --output-dir outputs/stylegan_lite --epochs 50 --batch-size 64
python compare_models.py --dcgan-checkpoint outputs/dcgan/checkpoints/latest.pt --stylegan-checkpoint outputs/stylegan_lite/checkpoints/latest.pt --num-images 5000 --batch-size 64
```

报告中可从 `outputs/metrics/model_comparison.csv` 引用以下指标：

| 模型 | 参数量 | 生成速度 images/s | IS mean | IS std | 观察结论 |
| --- | --- | --- | --- | --- | --- |
| DCGAN | 运行后填写 | 运行后填写 | 运行后填写 | 运行后填写 | 基础卷积上采样，训练快，细节表达能力有限 |
| StyleGAN-Lite | 运行后填写 | 运行后填写 | 运行后填写 | 运行后填写 | 风格调制能力更强，通常参数更多、生成更慢 |
