# week7_AI_3_GenerativeAI
- Generative AI
  - Image|Text|Music|Movie|Anything(X) generation 
- AutoEncoders(AE) and Variational autoencoder
- GAN(Generative Adversarial Networks)生成對抗網路
  - [GANs implementation using MNIST data](https://github.com/MorvanZhou/mnistGANs)

# 
- [Easiest way to download kaggle data in Google Colab](https://www.kaggle.com/general/74235)
- [How to Load Kaggle Datasets Directly Into Google Colab?](https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/)
# Generative AI
- [Generating Sound with Neural Network](https://www.youtube.com/playlist?list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp) [GITHUB](https://github.com/musikalkemist/generating-sound-with-neural-networks)
- [ChatGPT is not all you need. A State of the Art Review of large Generative AI models(202301)](https://arxiv.org/abs/2301.04655)
  - texts to images: DALLE-2 model
  - text to 3D images: Dreamfusion model
  - images to text: Flamingo model
  - texts to video:e Phenaki model
  - texts to audio: AudioLM model 
  - texts to other texts: ChatGPT
  - texts to code: Codex model
  - texts to scientific texts:Galactica model or even create algorithms like AlphaTensor. 
  
  
## AutoEncoders(AE) and Variational autoencoder
- [From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
- AutoEncoders(AE)
  - 【TensorFlow 官方教學課程】[Intro to Autoencoders](https://www.tensorflow.org/tutorials/generative/autoencoder)
  - 台大李弘毅教授演講[【機器學習2021】自編碼器 (Auto-encoder) (上) – 基本概念](https://www.youtube.com/watch?v=3oHlf8-J3Nc)
  - 台大李弘毅教授演講[ML Lecture 16: Unsupervised Learning - Auto-encoder](https://www.youtube.com/watch?v=Tk5B4seA-AU&t=472s)
- Denoising Autoencoders (DAE) 2008 去除雜訊
  - [Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) 
  - [](https://towardsdatascience.com/denoising-autoencoders-dae-how-to-use-neural-networks-to-clean-up-your-data-cd9c19bc6915)
- Variational autoencoder(VAE) 2013
  - 經典論文[Auto-Encoding Variational Bayes| Diederik P Kingma, Max Welling](https://arxiv.org/abs/1312.6114)
  - 陳縕儂 Vivian NTU MiuLab[台大資訊 深度學習之應用 | ADL 16.3: Variational Auto-Encoder (VAE) 固定特徵的分布資訊](https://www.youtube.com/watch?v=cjjjhHIDjKo)
  - 【TensorFlow 官方教學課程】[Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae) 
  - [An Introduction to Variational Autoencoders(2019)Diederik P. Kingma, Max Welling](https://arxiv.org/abs/1906.02691)
- k-Sparse Autoencoder 2013
- Contractive Autoencoder 2011


## GAN(Generative Adversarial Networks)生成對抗網路
- [參考書籍](./REF_GAN.md) [The GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)
- Generative Adversarial Networks by Ian Goodfellow [Generative Adversarial Networks (NIPS 2016 tutorial)](https://www.youtube.com/watch?v=HGYYEUSm-0Q)
- Tutorial on Generative Adversarial Networks by Mark Chang [Video](https://www.youtube.com/playlist?list=PLeeHDpwX2Kj5Ugx6c9EfDLDojuQxnmxmU)
- Deep Diving into GANs: From Theory to Production (EuroSciPy 2018) by Michele De Simoni, Paolo Galeone [Video](https://www.youtube.com/watch?v=CePrdabdtxw)
- [Generative Adversarial Networks | Data Science Summer School 2022](https://www.youtube.com/watch?v=xMJTylr4E30&t=2s)
## GAN(Generative Adversarial Networks)生成對抗網路 重要歷史發展
- 2014開山巨作 [Generative Adversarial Networks|Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio](https://arxiv.org/abs/1406.2661)
- CGan(Conditional Generative Adversarial Nets) 2014
  - [Conditional Generative Adversarial Nets Mehdi Mirza, Simon Osindero]()
- Deep Convolutional GANs(DCGAN) 2015
  - 論文[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks|Alec Radford, Luke Metz, Soumith Chintala](https://arxiv.org/abs/1511.06434) 
  - Tensorflow 實作
    - 【TensorFlow 官方教學課程】[Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/tutorials/generative/dcgan) 
  - Pytorch 實作
    - [Mastering PyTorch(2021)](https://www.packtpub.com/product/mastering-pytorch/9781789614381) [GITHUB](https://github.com/PacktPublishing/Mastering-PyTorch)Chapter 8: Deep Convolutional GANs 
- pix2pix 2016
  - 論文[Image-to-Image Translation with `Conditional Adversarial Networks`(條件式對抗網路)](https://arxiv.org/abs/1611.07004)
  - 【TensorFlow 官方教學課程】[pix2pix: Image-to-image translation with a conditional GAN](https://www.tensorflow.org/tutorials/generative/pix2pix)
- PGGAN漸近增長生成對抗網路
- CycleGAN 2017
  - UC Berkeley 論文[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
  - [官方網站](https://junyanz.github.io/CycleGAN/)
  - 【TensorFlow 官方教學課程】[CycleGAN](https://www.tensorflow.org/tutorials/generative/cyclegan)
  - Pytorch 實作 [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb)
  - 論文[Contrastive Learning for Unpaired Image-to-Image Translation(2020)](https://arxiv.org/abs/2007.15651)
- StackGAN(Stacked Generative Adversarial Networks)
  - 論文[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1612.03242) 
- StyleGAN
- SRGAN
- ColorGAN 2019
  - 論文[End-to-End Conditional GAN-based Architectures for Image Colourisation](https://arxiv.org/abs/1908.09873)
  - [官方GITHUB](https://github.com/bbc/ColorGAN)

## GAN應用
- [Generative Adversarial Networks for Malware Detection: a Survey](https://arxiv.org/search/?query=Generative+Adversarial+Networks&searchtype=all&source=header)
