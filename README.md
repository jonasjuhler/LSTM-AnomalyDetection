# LVEAD-02460
This repo contains an implementation of the LVEAD algorithm for anomaly detection described in the article [Time Series Anomaly Detection with Variational Autoencoders](https://arxiv.org/pdf/1907.01702.pdf).


# Github repos with examples of VAE for anomaly detection
  [This](https://github.com/ldeecke/vae-torch) repo has implemented a Torch CNN-based VAE to detect outliers in images.
  [Here](https://github.com/SeldonIO/seldon-core/blob/master/components/outlier-detection/vae) is a Keras implementation of an outlier detection VAE with simple dense layers (FNN)
  [This](https://github.com/JGuymont/vae-anomaly-detector) repo has a similar approach to the one above only with a PyTorch implementation instead of Keras.
  [This](https://github.com/KDD-OpenSource/DeepADoTS) repo might just be the most useful. It contains PyTorch implementations of a wide range of time series outlier detection algorithm (e.g. an LSTM based encoder-decoder structure).
  
