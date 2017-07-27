# sketch_rnn_classification
This repository contains code to classify sketches from the QuickDraw dataset provided by Google - https://github.com/googlecreativelab/quickdraw-dataset

This code utilizes the same structure as sketch_rnn code provided by Google here - https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn with a few modifications. This model is built by replacing the decoder with a hidden linear layer and replacing the reconstruction and KL divergence losses by cross entropy loss for classification. The code works with the magenta environment.

This code was tested on the following 16 classes from the dataset - moon.npz,mountain.npz,rain.npz,sun.npz,tree.npz,castle.npz,garden.npz,house.npz,key.npz,teddy_bear.npz,monkey.npz,mermaid.npz,duck.npz,dog.npz,bicycle.npz,angel.npz. This code achieves an accuracy of 92% on these 16 classes. More detailed results will be added shortly.
