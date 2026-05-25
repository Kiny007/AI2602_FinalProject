# project.pdf 要求摘要

任务：基于 GAN 的人头图像生成。

## 基本要求

1. 实现基础 GAN 模型，例如 DCGAN，并用于人头图像生成。
2. 在选定数据集上训练模型，生成较高质量的人头图像。
3. 测试两张人头图像之间线性插值的一系列结果。
4. 使用 FID 或 IS 评估生成图像质量。

## 当前项目对应实现

1. `src/gan_faces/models.py`：DCGAN 生成器和判别器。
2. `train.py`：训练入口，支持本地图片文件夹、LFW、CelebA。
3. `generate.py`：加载训练权重并生成图片网格。
4. `interpolate.py`：在两个潜变量端点之间线性插值，生成连续头像变化图。
5. `evaluate.py` 和 `src/gan_faces/metrics.py`：使用 Inception Score 评估生成质量。
