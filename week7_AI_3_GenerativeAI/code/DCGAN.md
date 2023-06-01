# 說明
- 本教程演示了如何使用[深度卷積生成對抗網路](https://arxiv.org/pdf/1511.06434.pdf) (DCGAN) 生成手寫數位的圖像。
- 該代碼是使用 [Keras 序列式 API](https://tensorflow.google.cn/guide/keras) 與 `tf.GradientTape` 訓練迴圈編寫的。

## 什麼是生成對抗網路？
- [生成對抗網路](https://arxiv.org/abs/1406.2661) (GAN) 是當今電腦科學領域最有趣的想法之一。
- 兩個模型通過對抗過程同時訓練。
- *生成器*（“藝術家”）學習創建看起來真實的圖像，
- 而*判別器*（“藝術評論家”）學習區分真假圖像。

![生成器和判別器圖示](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gan1.png?raw=1)

- 訓練過程中，*生成器*在生成逼真圖像方面逐漸變強，而*判別器*在辨別這些圖像的能力上逐漸變強。
- 當*判別器*不再能夠區分真實圖片和偽造圖片時，訓練過程達到平衡。

![生成器和判別器圖示二](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gan2.png?raw=1)

本筆記在 MNIST 資料集上演示了該過程。下方動畫展示了當訓練了 50 個epoch （全部資料集反覆運算50次） 時*生成器*所生成的一系列圖片。

圖片從隨機雜訊開始，隨著時間的推移越來越像手寫數字。

![輸出樣本](https://tensorflow.google.cn/images/gan/dcgan.gif)

# 安裝產生 GIFs的套件
```
!pip install imageio
!pip install git+https://github.com/tensorflow/docs
```
### Import TensorFlow and other libraries
```

import tensorflow as tf

tf.__version__
```

```
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display
```
### 載入和準備資料集
- 使用 MNIST 資料集來訓練生成器和判別器。生成器將生成類似於 MNIST 資料集的手寫數字。
```
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

```
## 創建模型

生成器和判別器均使用 [Keras Sequential API](https://tensorflow.google.cn/guide/keras#sequential_model) 定義。

### 生成器
```
生成器使用 `tf.keras.layers.Conv2DTranspose`（上採樣）層來從種子（隨機雜訊）中生成圖像。
以一個使用該種子作為輸入的 `Dense` 層開始，然後多次上採樣，直至達到所需的 28x28x1 的圖像大小。
請注意，除了輸出層使用雙曲正切之外，其他每層均使用 `tf.keras.layers.LeakyReLU` 啟動。
```
```
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
```
"""使用（尚未訓練的）生成器創建一張圖片。"""
```
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
```
"""
### 判別器:判別器是一個基於 CNN 的圖片分類器。
```
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
```
"""使用（尚未訓練的）判別器對所生成的圖像進行真偽分類。模型將被訓練為對真實圖像輸出正值，對偽造圖像輸出負值。"""
```
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)
```
## 定義損失函數和優化器

為兩個模型定義損失函數和優化器。

# This method returns a helper function to compute cross entropy loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

### 判別器損失
- 該方法量化判別器從判斷真偽圖片的能力。
- 它將判別器對真實圖片的預測值與值全為 1 的陣列進行對比，
- 將判別器對偽造（生成的）圖片的預測值與值全為 0 的陣列進行對比。

```

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
```

### 生成器損失
- 生成器的損失可量化其欺騙判別器的能力。
- 直觀地說，如果生成器表現良好，判別器會將偽造圖像分類為真實圖像（或 1）。
- 在此，需要將判別器對生成圖像的決策與值全為 1 的陣列進行對比。
```
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

"""判別器和生成器優化器不同，因為您將分別訓練兩個網路。"""

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
```

### 保存檢查點
- 本筆記還演示了如何保存和恢復模型，這在長時間訓練任務被中斷的情況下比較有説明。
```
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
```
## 定義訓練迴圈
```
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

"""訓練迴圈在生成器接收到一個隨機種子作為輸入時開始。該種子用於生成一個圖像。判別器隨後被用於對真實圖像（選自訓練集）和偽造圖像（由生成器生成）進行分類。為每一個模型計算損失，並使用梯度更新生成器和判別器。"""

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
```

### 用來生成與保存圖片的函數
```
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
```

## 訓練模型
- 上面定義的 `train()` 方法來同時訓練生成器和判別器。
- 訓練 GANs 可能是棘手的。
- 重要的是，生成器和判別器不能夠互相壓制對方（例如，他們以相似的學習率訓練）。
- 在訓練之初，生成的圖片看起來像是隨機雜訊。
- 隨著訓練過程的進行，生成的數位將越來越真實。
- 在大概 50 個 epoch 之後，這些圖片看起來像是 MNIST 數位。
- 使用 Colab 中的預設設置可能需要大約 1 分鐘每 epoch。
```
train(train_dataset, EPOCHS)

"""恢復最新的檢查點。"""

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```
## 創建 GIF


### Display a single image using the epoch number
````
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

"""使用訓練過程中生成的圖片通過 `imageio` 生成動態 gif"""

anim_file = 'dcgan.gif'

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
```
