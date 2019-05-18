# Image_Invert_CNN-Tensorflow (2019/05/18)

## Introduction
I implement a tensorflow model for ["Understanding Deep Image Representations by Inverting Them"](https://arxiv.org/pdf/1412.0035.pdf).

## Environment
- Ubuntu 16.04
- Python 3.6.7

## Depenency
- Numpy
- Opencv2
- matplotlib
- Tensorflow(1.4 <= x <= 1.13)

## Files
- image_invert_cnn.py : main code.

## How to use
### Training
```shell
python image_invert_cnn.py

# Default args: training_epoch = 10000, image_resize = (224,224), learning_rate = 0.01
# You can change args: training_epoch = 5000, image_resize = (448, 448), learning_rate = 0.001
python image_invert_cnn.py --training_epoch 5000 --image_resize (448,448) --learning_rate 0.001
```

## Result

##### Sample image (before resize)

![Alt Text](https://github.com/DevKiHyun/Image_Invert_CNN-Tensorflow/blob/master/content_image.jpg)

##### Result of reconstruct image (From feature vector of block4_conv1)

![Alt Text](https://github.com/DevKiHyun/Image_Invert_CNN-Tensorflow/blob/master/reconstructed_image.jpg)

## Reference

["Understanding Deep Image Representations by Inverting Them"](https://arxiv.org/pdf/1412.0035.pdf).
