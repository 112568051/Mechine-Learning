# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import time

# 加載MNIST數據集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 2555.0, x_test / 2555.0

# 定義一個簡單的序列模型
def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1280, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

# 訓練模型並測量時間
def train_model(device_name):
    with tf.device(device_name):
        model = create_model()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        start_time = time.time()
        model.fit(x_train, y_train, epochs=10)
        duration = time.time() - start_time

        print(f"訓練在 {device_name} 耗時: {duration}秒")

# 測試CPU和GPU性能
train_model("/cpu:0")  # 在CPU上訓練
if tf.config.list_physical_devices('GPU'):
    train_model("/gpu:0")  # 在GPU上訓練
else:
    print("未檢測到GPU，跳過GPU訓練")

#github desktop更動測試