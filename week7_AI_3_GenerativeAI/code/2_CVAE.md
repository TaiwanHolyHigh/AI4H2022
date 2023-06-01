# 卷積變分自編碼器



此筆記本演示了如何基於 MNIST 資料集訓練變分自編碼器 (VAE) ([1](https://arxiv.org/abs/1312.6114), [2](https://arxiv.org/abs/1401.4082))。VAE 是一種自編碼器取值的概率分佈，該模型會獲取高維輸入資料並將其壓縮為較小的表示。與將輸入映射到隱向量的傳統自編碼器不同，VAE 會將輸入資料映射到概率分佈的參數中，例如高斯分佈的均值和方差。這種方式可以生成一個連續、結構化的隱空間，對於圖像生成而言十分適用。

![CVAE 圖像隱空間](https://gitlocalize.com/repo/4592/zh-cn/site/en-snapshot/tutorials/generative/images/cvae_latent_space.jpg)

## 導入 Tensorflow 與其他庫
```

!pip install tensorflow-probability

# to generate gifs
!pip install imageio
!pip install git+https://github.com/tensorflow/docs

from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

```
## 載入 MNIST 資料集

每個 MNIST 圖像最初都是一個由 784 個整數組成的向量，每個整數在 0-255 之間，代表一個圖元的強度。在我們的模型中使用伯努利分佈對每個圖元進行建模，並對資料集進行靜態二值化。
```

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

```## 使用 *tf.data* 來將數據分批和打亂```

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

```## 通過 *tf.keras.Sequential* 連接生成網路與推理網路

在此 VAE 示例中，對編碼器和解碼器網路使用兩個小型 ConvNet。在文獻中，這些網路也分別稱為推斷/識別和生成模型。使用 `tf.keras.Sequential` 來簡化實現。在下面的描述中，使 $x$ 和 $z$ 分別表示觀測值和隱變數。

### 生成網路

這定義了近似後驗分佈 $q(z|x)$，它會將輸入取作觀測值並輸出一組參數，用於指定隱變數表示 $z$ 的條件分佈。在本例中，簡單地將分佈建模為對角高斯分佈，網路會輸出分解高斯分佈的均值和對數方差參數。輸出對數方差而不是直接用於數值穩定性的方差。

### 推理網路

這定義了觀測值的條件分佈 $p(x|z)$，它會將隱變數樣本 $z$ 取作輸入並輸出觀測值條件分佈的參數。將隱變數先驗分佈 $p(z)$ 建模為單位高斯分佈。

### 重參數化技巧

要在訓練期間為解碼器生成樣本 $z$，您可以在給定輸入觀測值 $x$ 的情況下從編碼器輸出的參數所定義的隱變數分佈中採樣。然而，這種採樣操作會產生瓶頸，因為反向傳播不能流經隨機節點。

要解決這個問題，請使用重參數化技巧。在我們的示例中，使用解碼器參數和另一個參數 $\epsilon$ 來逼近 $z$，如下所示：

$$z = \mu + \sigma \odot \epsilon$$

其中 $\mu$ 和 $\sigma$ 分別代表高斯分佈的均值和標準差。它們可通過解碼器輸出推導得出。$\epsilon$ 可被認為是用於保持 $z$ 的隨機性的隨機雜訊。從標準正態分佈生成 $\epsilon$。

隱變數 $z$ 現在由 $\mu$、$\sigma$ 和 $\epsilon$ 的函數生成，這將使模型能夠分別通過 $\mu$ 和 $\sigma$ 在編碼器中反向傳播梯度，同時通過 $\epsilon$ 保持隨機性。

### 網路架構

對於編碼器網路，使用兩個卷積層後接一個全連接層。在解碼器網路中，通過使用一個全連接層後接三個卷積轉置層（在某些背景下也稱為反卷積層）來鏡像此架構。請注意，通常的做法是在訓練 VAE 時避免使用批量歸一化，因為使用 mini-batch 導致的額外隨機性可能會在提高採樣隨機性的同時加劇不穩定性。

```

class CVAE(tf.keras.Model):
  ```Convolutional variational autoencoder.```

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

```## 定義損失函數和優化器

VAE 通過最大化邊際對數似然的證據下界（ELBO）進行訓練：

$$\log p(x) \ge \text{ELBO} = \mathbb{E}_{q(z|x)}\left[\log \frac{p(x, z)}{q(z|x)}\right].$$

在實踐中，優化此期望的單樣本蒙特卡羅估值：

$$\log p(x| z) + \log p(z) - \log q(z|x)$$，其中 $z$ 從 $q(z|x)$ 中採樣。

注：您也可以分析計算 KL 項，但為了簡單起見，您在此處會將三個項全部應用到蒙特卡羅 Estimator 中。
```

optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  ```Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  ```
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

```## 訓練

- 首先反覆運算資料集
- 在每次反覆運算期間，將圖像傳遞給編碼器以獲取近似後驗分佈 $q(z|x)$ 的一組均值和對數方差參數
- 然後，應用*重參數化技巧*以從 $q(z|x)$ 中採樣
- 最後，將重參數化的樣本傳遞給解碼器以獲取生成分佈 $p(x|z)$ 的 logit
- 注：由於您使用由 Keras 載入的資料集，訓練集中有 6 萬個資料點，測試集中有 1 萬個資料點，因此我們基於測試集得出的 ELBO 會略高於使用 Larochelle 的 MNIST 動態二值化的文獻中報告的結果。

### 生成圖像

- 進行訓練後，可以生成一些圖片了
- 首先從單位高斯先驗分佈 $p(z)$ 中採樣一組隱向量
- 隨後生成器將潛在樣本 $z$ 轉換為觀測值的 logit，得到分佈 $p(x|z)$
- 在此處，繪製伯努利分佈的概率分佈圖

```

epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]

generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()

  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  display.clear_output(wait=False)
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
        .format(epoch, elbo, end_time - start_time))
  generate_and_save_images(model, epoch, test_sample)

```### 使用 epoch 編號顯示圖片```

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

plt.imshow(display_image(epoch))
plt.axis('off')  # Display images

```### 生成所有保存圖片的 GIF```

anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)

```### 顯示隱空間中數位的二維流形

運行下面的代碼將顯示不同數位類的連續分佈，每個數位都會在二維隱空間中變形為另一數位。使用 [TensorFlow Probability](https://tensorflow.google.cn/probability) 為隱空間生成標準正態分佈。
```

def plot_latent_images(model, n, digit_size=28):
  ```Plots n x n digit images decoded from the latent space.```

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      x_decoded = model.sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.show()

plot_latent_images(model, 20)

```

## 後續步驟

本教程演示了使用 TensorFlow 實現卷積變分自編碼器的方式。

下一步，您可以嘗試通過增大網路來改進模型輸出。例如，您可以嘗試將每個 `Conv2D` 和 `Conv2DTranspose` 層的 `filter` 參數設置為 512。請注意，為了生成最終的二維隱空間圖像，您需要將 `latent_dim` 保持為 2。此外，訓練時間會隨網路的增大而延長。

您還可以嘗試使用不同的資料集實現 VAE，例如 CIFAR-10。

VAE 支持以多種不同的風格和不同的複雜性實現。您可以從以下資源中找到其他實現：

- [變分自編碼器 (keras.io)](https://keras.io/examples/generative/vae/)
- [“編寫自訂層和模型”指南中的 VAE 示例 (tensorflow.org)](https://tensorflow.google.cn/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example)
- [TFP 概率層：變分自動編碼器](https://tensorflow.google.cn/probability/examples/Probabilistic_Layers_VAE)

```
