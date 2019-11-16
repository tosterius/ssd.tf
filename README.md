## Yet another SSD implementation


This repository contains a pretty straightforward implementation of the SSD algorithm.
The code is well-commented. In some tricky places, there are references 
to the concrete paragraph of the [SSD report](https://arxiv.org/pdf/1512.02325).

### Requirements
 1) Python 3.4+
 2) Tensorflow 1.13+
 3) opencv 3.4+
 4) [VGG16](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) (if you want to train the model on
 your own data)
 
### Training
 To train the model on your own dataset you will need to extend Dataset
 class. The only thing it does is reading and parsing the dataset. See
 VocDataset as an example.

###  