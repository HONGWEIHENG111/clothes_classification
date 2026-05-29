## Fashion MNIST 深度图像分类器
基于 TensorFlow 和 Keras 构建的深度卷积神经网络（CNN），用于对 Fashion MNIST 数据集中的 10 类服饰图像进行高精度分类。本项目不仅包含了完整的模型构建与训练流程，还集成了丰富的评估指标和直观的数据可视化功能。

## 🌟 核心特性
现代 CNN 架构：采用连续卷积块设计，结合了 BatchNormalization 和性能更优的 Swish 激活函数。使用 GlobalAveragePooling2D 替代传统的展平操作（Flatten），有效减少参数量并降低过拟合风险。
自适应训练策略：

早停机制 (Early Stopping)：监控验证集 Loss，在连续 8 个 Epoch 无提升时自动停止训练，并恢复最佳权重。

学习率衰减 (ReduceLROnPlateau)：当模型陷入平台期时，自动将学习率减半，帮助模型跳出局部最优。

深度性能评估：除了基础的 Accuracy 之外，还使用 scikit-learn 和 pandas 计算并展示了 F1 Score (Macro)、Recall (Macro) 以及包含各项细节的分类报告 (Classification Report)。

混淆矩阵与可视化：自动计算混淆矩阵并将其绘制为高质量图像 (confusion_matrix.png) 保存至本地；同时在训练结束后弹出直观的预测概率条形图对比。

硬件自适应：程序启动时会自动检测 GPU 环境，并开启显存的按需分配（Memory Growth），防止 TensorFlow 独占全部显存。

## 🛠️ 环境依赖
在运行代码之前，请确保您的环境中已安装以下 Python 库：

请在终端中运行以下命令来安装所需的环境：

```bash
pip install -r requirements.txt
```

或者手动安装指定库：

```bash
pip install numpy matplotlib tensorflow scikit-learn pandas
```

## 运行项目

安装完依赖后，执行以下命令即可启动训练：

```bash
python main.py
```
程序运行流程：

检测 GPU 状态。

自动下载并预处理 Fashion MNIST 数据集（归一化至 0-1 并扩展通道维度）。

构建模型并开始训练（默认 50 个 Epoch，验证集划分 20%）。

训练结束后，在终端打印评估指标和混淆矩阵 DataFrame。

在当前目录下生成并保存 confusion_matrix.png。

弹出可视化窗口，展示部分测试集图片的预测结果与真实标签对比。

## 🧠 模型结构简述
输入尺寸: (28, 28, 1)

Block 1 (基础特征提取): 2x [Conv2D(32) -> BatchNorm -> Swish] -> MaxPool -> Dropout(0.25)

Block 2 (中级特征提取): 2x [Conv2D(64) -> BatchNorm -> Swish] -> MaxPool -> Dropout(0.3)

Block 3 (高阶细粒度特征): Conv2D(128) -> BatchNorm -> Swish

Classification Head: GlobalAveragePooling2D -> Dense(256) -> BatchNorm -> ReLU -> Dropout(0.5) -> Dense(10, Linear/Logits)

## 📊 输出说明
运行结束后，您将获得以下结果输出：

终端日志：每个 Epoch 的训练/验证 Loss 和 Accuracy。

详细测试指标：整体 Test Accuracy、Macro Recall、Macro F1 Score。

分类报告：涵盖 10 个类别（如 T-shirt/top, Trouser, Shirt 等）的精准度（Precision）、召回率（Recall）和 F1 值。

图表输出：

本地保存的 confusion_matrix.png：用于分析模型在哪些相似类别（如 Shirt 与 T-shirt/top）上容易发生混淆。

弹出的 matplotlib 窗口：直观展示单张图片的预测结果分布。

## ⚙️ 超参数配置
您可以直接在代码顶部的全局变量区修改基础超参数以进行调优：

Python

EPOCHS = 50             # 训练轮数

BATCH_SIZE = 32         # 批次大小

HIDDEN_UNITS = 128      # 隐藏层神经元（可按需扩展）

VALIDATION_SPLIT = 0.2  # 验证集比例