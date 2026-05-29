# project.pdf 要求摘要

任务：基于 GAN 的人头图像生成。

## 基本要求

1. 实现基础 GAN 模型，例如 DCGAN，并用于人头图像生成。
2. 在选定数据集上训练模型，生成较高质量的人头图像。
3. 测试两张人头图像之间线性插值的一系列结果。
4. 使用 FID 或 IS 评估生成图像质量。

## 当前项目对应实现

1. `src/gan_faces/models.py`：DCGAN 生成器、轻量 StyleGAN 风格生成器、CycleGAN 生成器和判别器。
2. `train.py`：训练入口，支持本地图片文件夹、LFW、CelebA。
3. `generate.py`：加载训练权重并生成图片网格。
4. `interpolate.py`：在两个潜变量端点之间线性插值，生成连续头像变化图。
5. `evaluate.py` 和 `src/gan_faces/metrics.py`：使用 Inception Score 评估生成质量。
6. `train_stylegan.py`：训练轻量 StyleGAN 风格生成器，用于改进模型实验。
7. `compare_models.py`：比较 DCGAN 与 StyleGAN-Lite 的参数量、生成速度和 IS。
8. `train_cyclegan.py`：训练 CycleGAN，用于两个无配对图像域之间的转换实验。
9. `compare_gan_cyclegan.py`：比较 DCGAN 与 CycleGAN 的参数量、生成速度、IS 和循环重建误差。

## Bonus 1 对应实现

Bonus 1 要求：对比改进的 GAN 模型（如 StyleGAN 或 CycleGAN）与基础模型的性能差异。

当前项目支持两条 Bonus 对比路线。

### 路线一：StyleGAN-Lite

`StyleGeneratorLite` 保留了 StyleGAN 的几个核心组件：

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

### 路线二：CycleGAN

如果报告选择 CycleGAN，需要准备两个无配对图像域，例如普通头像域 A 和目标风格头像域 B。新增实现包括：

1. `CycleGenerator`：ResNet 风格图像到图像翻译生成器。
2. `PatchDiscriminator`：PatchGAN 判别器，判断局部图像块是否真实。
3. 循环一致性损失：约束 `A -> B -> A` 和 `B -> A -> B` 重建原图。
4. 身份保持损失：让目标域图片输入目标方向生成器时尽量保持不变。

建议实验命令：

```powershell
python train.py --dataset folder --data-root data/domain_b --output-dir outputs/dcgan --epochs 50 --batch-size 64
python train_cyclegan.py --domain-a-root data/domain_a --domain-b-root data/domain_b --output-dir outputs/cyclegan --epochs 50 --batch-size 4
python compare_gan_cyclegan.py --dcgan-checkpoint outputs/dcgan/checkpoints/latest.pt --cyclegan-checkpoint outputs/cyclegan/checkpoints/latest.pt --domain-a-root data/domain_a --domain-b-root data/domain_b --direction a2b --num-images 1000 --batch-size 64
```

报告中可从 `outputs/metrics/gan_vs_cyclegan.csv` 引用以下指标：

| 模型 | 输入形式 | 生成器参数量 | 完整模型参数量 | 生成速度 images/s | IS mean | IS std | cycle_l1 | 观察结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DCGAN | 随机噪声 | 运行后填写 | 运行后填写 | 运行后填写 | 运行后填写 | 运行后填写 | 不适用 | 无条件生成，速度快，但不控制输入身份或风格转换 |
| CycleGAN A->B | A 域图片 | 运行后填写 | 运行后填写 | 运行后填写 | 运行后填写 | 运行后填写 | 运行后填写 | 支持无配对域转换，但模型更重、训练更慢 |

注意：DCGAN 和 CycleGAN 的任务定义不同。DCGAN 学习目标域图像分布并从噪声采样；CycleGAN 学习两个图像域之间的映射。二者的 IS 和速度可以作为性能参考，但 CycleGAN 的 `cycle_l1` 只能用于评价循环一致性，不能直接与 DCGAN 对应。
