Fashion MNIST 图像分类与深度评估 (CNN)
本项目是一个基于 TensorFlow 2.x 构建的卷积神经网络 (CNN) 图像分类器，专门针对 Fashion MNIST 服装数据集进行训练与预测。

除了基础的模型训练，本项目还整合了 Scikit-learn 的深度评估指标（如混淆矩阵、F1 Score、Recall）以及 Matplotlib 的直观可视化功能，为你提供从模型构建、训练、早停控制到多维度性能评估的端到端完整流线。

🌟 核心特性
现代化 CNN 架构：引入了 BatchNormalization（批量归一化）和 Dropout，有效加速收敛并防止模型过拟合。

智能显存管理：自动检测可用 GPU，并开启显存按需分配（Memory Growth），避免 TensorFlow 独占全部显存。

自适应训练 (Early Stopping)：监控验证集 Loss，当模型在连续 5 个 Epoch 内不再提升时自动停止训练，并恢复到表现最优的权重。

深度性能评估：不局限于简单的准确率 (Accuracy)，额外计算并输出 宏平均 (Macro) F1 Score 和 Recall，并生成详细的分类报告。

丰富的数据可视化：

自动绘制、保存并在程序末尾自动弹出高清的混淆矩阵图像 (confusion_matrix.png)。

生成包含 15 张测试集图片的预测概率分布条形图。

🛠️ 环境依赖
在运行本项目之前，请确保你的环境中安装了以下 Python 库。推荐使用 Python 3.8 或以上版本。

```bash
pip install tensorflow numpy matplotlib scikit-learn
```
TensorFlow: 用于构建和训练深度学习模型。

NumPy: 用于矩阵计算和数据预处理。

Matplotlib: 用于绘制预测结果和混淆矩阵图像。

Scikit-learn: 用于计算高级评估指标（分类报告、F1、Recall、混淆矩阵）。

🧠 模型架构说明
本项目没有采用简单的全连接网络，而是构建了一个两层卷积块组成的 CNN 架构，模型预测时，代码额外封装了一个包含 Softmax 层的概率模型 (probability_model)，以输出 0~1 之间的类概率。

🚀 快速开始
克隆或下载本代码后，在终端中直接运行主程序即可：

```Bash
python main.py
```
运行流程概览：
初始化：打印 TensorFlow 版本，检测 GPU 状态并配置显存。

数据加载：自动下载并预处理 Fashion MNIST 数据集。

模型训练：根据设定的超参数（默认 10 个 Epochs，Batch Size 32，20% 验证集）开始训练，并触发 EarlyStopping。

指标评估：在终端打印 Test Accuracy、混淆矩阵、Macro F1、Macro Recall 及详细分类报告。

图像保存：在当前目录下生成高清的 confusion_matrix.png。

可视化展示：弹出一个 Matplotlib 窗口，直观展示 15 个测试样本的图片及其预测概率分布。

⚙️ 超参数配置 (Hyperparameters)
你可以在 main.py 的文件顶部轻松修改以下超参数，以探索不同配置对模型性能的影响：

EPOCHS = 10 (最大训练轮数)

BATCH_SIZE = 32 (批次大小)

HIDDEN_UNITS = 128 (全连接层神经元数量)

VALIDATION_SPLIT = 0.2 (用于验证集的数据比例)

💡 小贴士： 程序运行到最后会同时弹出两个 Matplotlib 图片窗口。程序会暂停运行直至你手动关闭这两个窗口。