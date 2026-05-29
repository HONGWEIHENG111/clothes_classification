import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, ConfusionMatrixDisplay
import pandas as pd
# --- 超参数配置 ---
EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_UNITS = 128
VALIDATION_SPLIT = 0.2

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 开启 GPU 显存按需分配
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"【系统通知】检测到 GPU 正在运行！可用 GPU 数量: {len(gpus)}")
    for gpu in gpus:
        print(f" -> GPU 设备: {gpu.name}")
    try:
        # 开启显存按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("【系统通知】未检测到 GPU，程序将默认使用 CPU 运行。")


def plot_image(i, predictions_array, true_label, img):
    """绘图辅助函数：绘制单张图片及其预测结果"""
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.squeeze(), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    # 使用 f-string 提升可读性
    plt.xlabel(f"{CLASS_NAMES[predicted_label]} {100 * np.max(predictions_array):2.0f}% ({CLASS_NAMES[true_label]})", color=color)


def plot_value_array(i, predictions_array, true_label):
    """绘图辅助函数：绘制预测概率的条形图"""
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def main():
    print(f"TensorFlow Version: {tf.__version__}")

    print("正在加载并预处理数据集...")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # 归一化
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # 为 CNN 增加通道维度，使其形状变为 (28, 28, 1)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # 构建与编译模型
    # 使用 CNN 架构
    # 数据增强
    #data_augmentation = tf.keras.Sequential([
        #tf.keras.layers.RandomFlip("horizontal", input_shape=(28, 28, 1)),  # 随机水平翻转
        #tf.keras.layers.RandomRotation(0.05),  # 随机旋转约 18 度 (0.05 * 360)
        #tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # 随机缩放 10%
        #tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # 随机平移 10%
    #])
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
        #data_augmentation,

        # 第一层卷积块 (增加连续卷积，提升非线性表达)
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('swish'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('swish'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        # 第二层卷积块
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('swish'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('swish'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),  # 稍微增加Dropout防止深层网络过拟合

        # 第三层卷积块 (专门用于提取高阶细粒度特征，区分领口/纽扣)
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('swish'),

        # 展平与全连接层
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # 每次降低一半的学习率
        patience=3,  # 如果3个epoch没提升，就降学习率
        min_lr=1e-5,  # 学习率下限
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # 监控验证集的损失值
        patience=8,  # 如果连续 5 个 epoch 验证集 loss 都没有下降，则停止训练
        restore_best_weights=True  # 训练停止后，自动恢复到表现最好的那一次的模型权重
    )

    print("开始训练模型...")

    # 给易错类别（如 Shirt）增加权重，基础类别保持为 1.0
    custom_class_weight = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
        5: 1.0, 6: 1.05,  # 重点关注 Shirt
        7: 1.0, 8: 1.0, 9: 1.0
    }
    # 加入 validation_split 和 batch_size 增强训练过程的可见性和控制力
    model.fit(train_images,
              train_labels,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_split=VALIDATION_SPLIT,
              #class_weight=custom_class_weight,
              callbacks=[early_stopping, reduce_lr])
    print("\n评估模型性能...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')

    # 构建概率模型并预测
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)
    # 取出概率最大的索引作为预测类别
    predicted_classes = np.argmax(predictions, axis=1)

    print("\n--- 模型深度评估 ---")
    # 计算并打印混淆矩阵
    cm = confusion_matrix(test_labels, predicted_classes)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("混淆矩阵 (带标签):")
    # 设置 pandas 显示选项，防止列数太多被折叠
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(cm_df)

    # 绘制并保存混淆矩阵照片
    fig, ax = plt.subplots(num="Confusion Matrix", figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)  # 使用蓝色调，文字倾斜45度防止重叠
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("-> 混淆矩阵图像已保存为当前目录下的 'confusion_matrix.png'")

    # 计算 F1 和 Recall (使用 macro 平均)
    recall = recall_score(test_labels, predicted_classes, average='macro')
    f1 = f1_score(test_labels, predicted_classes, average='macro')
    print(f"\nRecall (Macro): {recall:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")

    # 打印详细分类报告（可选，非常推荐，能直接看到每个类的 F1 和 Recall）
    print("\n详细分类报告:")
    print(classification_report(test_labels, predicted_classes, target_names=CLASS_NAMES))
    # 绘制预测结果
    num_rows, num_cols = 5, 3
    num_images = num_rows * num_cols
    plt.figure("Prediction Results", figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()

    # 单张图片预测
    img = np.expand_dims(test_images[1], 0)
    predictions_single = probability_model.predict(img)
    print(f"单张图片预测完成，真实标签为: {CLASS_NAMES[test_labels[1]]}，预测标签为: {CLASS_NAMES[np.argmax(predictions_single[0])]}")

    print("所有任务完成。请手动关闭图片窗口以退出程序...")
    plt.show()

if __name__ == '__main__':
    main()