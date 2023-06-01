# 自編碼器簡介



本教程通過以下三個示例介紹自編碼器：基礎知識、圖像降噪和異常檢測。

自編碼器是一種特殊類型的神經網路，經過訓練後可將其輸入複製到其輸出。例如，給定一個手寫數位的圖像，自編碼器首先將圖像編碼為低維的潛在表示，然後將該潛在表示解碼回圖像。自編碼器學習壓縮資料，同時最大程度地減少重構誤差。

要詳細瞭解自編碼器，請考慮閱讀 Ian Goodfellow、Yoshua Bengio 和 Aaron Courville 撰寫的[《深度學習》](https://www.deeplearningbook.org/)一書的第 14 章。

## 導入 TensorFlow 和其他庫

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

## 載入資料集

- 使用 Fashion MNIST 資料集訓練基本自編碼器。
- 此資料集中的每個圖像均為 28x28 圖元。 
```

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)

```
## 第一個示例：基本自編碼器

![Basic autoencoder results](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/images/intro_autoencoder_result.png?raw=1)

定義一個具有兩個密集層的自編碼器：一個將圖像壓縮為 64 維隱向量的 `encoder`，以及一個從隱空間重構原始圖像的 `decoder`。

要定義模型，請使用 [Keras Model Subclassing API](https://tensorflow.google.cn/guide/keras/custom_layers_and_models)。

```

latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(784, activation='sigmoid'),
      layers.Reshape((28, 28))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
```
"""使用 `x_train` 作為輸入和目標來訓練模型。`encoder` 會學習將資料集從 784 個維度壓縮到隱空間，而 `decoder` 將學習重構原始圖像。"""
```
autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

"""現在，模型已經訓練完成，我們通過對測試集中的圖像進行編碼和解碼來測試該模型。"""

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
```
# 第二個示例：圖像降噪

![Image denoising results](https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/images/image_denoise_fmnist_results.png?raw=true)

經過訓練後，自編碼器還可以去除圖像中的噪點。在以下部分中，您將通過對每個圖像應用隨機雜訊來創建有噪版本的 Fashion MNIST 資料集。隨後，您將使用有噪圖像作為輸入並以原始圖像作為目標來訓練自編碼器。

我們重新導入資料集以忽略之前所做的修改：
```

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)

"""向圖像添加隨機雜訊："""

noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape) 

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

"""繪製有噪圖像：

"""

n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
plt.show()
```
### 定義卷積自編碼器

在此示例中，您將使用 `encoder` 中的 [Conv2D](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D) 層和 `decoder` 中的 [Conv2DTranspose](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2DTranspose) 層來訓練卷積自編碼器。
```

class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Denoise()

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

"""我們來看一下編碼器的摘要。請注意圖像是如何從 28x28 圖元下採樣為 7x7 圖元的。"""

autoencoder.encoder.summary()

"""解碼器將圖像從 7x7 圖元上採樣為 28x28 圖元。"""

autoencoder.decoder.summary()

"""繪製由自編碼器生成的有噪圖像和去噪圖像。"""

encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()

```
# 第三個示例：異常檢測

## 概述

在此示例中，您將訓練自編碼器來檢測 [ECG5000 資料集](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000)上的異常。此資料集包含 5,000 個[心電圖](https://en.wikipedia.org/wiki/Electrocardiography)，每個心電圖擁有 140 個資料點。您將使用簡化版的資料集，其中每個樣本都被標記為 `0`（對應於異常心律）或 `1`（對應于正常心律）。您需要關注如何識別異常心律。

注：這是一個有標籤的資料集，因此您可以將其表述為一個監督學習問題。此示例的目標是說明可應用於沒有可用標籤的大型資料集的異常檢測概念（例如，如果您有成千上萬個正常心律，而只有少量異常心律）。

您將如何使用自編碼器檢測異常？回想一下，自編碼器經過訓練後可最大程度地減少重構誤差。您將只基於正常心律訓練自編碼器，隨後使用它來重構所有資料。我們的假設是，異常心律存在更高的重構誤差。隨後，如果重構誤差超過固定閾值，則將心律分類為異常。

### 載入心電圖數據

您將使用的資料集基於 [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) 中的資料集。
```

# Download the dataset
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

"""將數據歸一化為 `[0,1]`。

"""

min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

"""您將僅使用正常心律訓練自編碼器，在此資料集中，正常心律被標記為 `1`。將正常心律與異常心律分開。"""

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

"""繪製正常的心電圖。 """

plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()

"""繪製異常的心電圖。"""

plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()
```
### 構建模型"""

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

"""請注意，自編碼器僅使用正常的心電圖進行訓練，但使用完整的測試集進行評估。"""

history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

"""如果重構誤差比正常訓練樣本大一個標準差，您可以快速地將心電圖歸類為異常。首先，我們繪製訓練集中的一個正常心電圖，隨後繪製自編碼器對其進行編碼和解碼後的重構以及重構誤差。"""

encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

"""創建一個類似的繪圖，這次是一個異常的測試樣本。"""

encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
```
### 檢測異常

通過計算重構損失是否大於固定閾值來檢測異常。在本教程中，您將計算訓練集中正常樣本的平均誤差，如果重構誤差比訓練集大一個標準差，則將未來的樣本分類為異常。

根據訓練集中的正常心電圖繪製重構誤差：
```
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

"""選擇一個比平均值高一個標準差的閾值。"""

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

"""注：還有其他可用來選擇閾值的策略，高於該閾值時，應將測試樣本分類為異常，正確的方式將取決於您的資料集。您可以通過本教程末尾的連結瞭解更多資訊。

如果檢查測試集中異常樣本的重構誤差，您會注意到大多數異常樣本的重構誤差都比閾值大。通過更改閾值，您可以調整分類器的[精確率](https://developers.google.com/machine-learning/glossary#precision)和[召回率](https://developers.google.com/machine-learning/glossary#recall)。
"""

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

"""如果重構誤差大於閾值，則將心電圖分類為異常。"""

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)

```
