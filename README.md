# sketch_rnn_classification
This repository contains code to classify sketches from the QuickDraw dataset provided by Google - https://github.com/googlecreativelab/quickdraw-dataset

This code utilizes the same structure as sketch_rnn code provided by Google here - https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn with a few modifications. This model is built by replacing the decoder with a hidden linear layer and replacing the reconstruction and KL divergence losses by cross entropy loss for classification. The code works with the magenta environment.

This code was tested on the following 16 classes from the dataset - moon, mountain, rain, sun, tree, castle, garden, house, key, teddy_bear, monkey, mermaid, duck, dog, bicycle, and angel. The training set for the experiment was created by combining the 70000 training sketches from each class. The validation set was created by combining the 2500 valodation sketches from each class. This code achieves an accuracy of 92% on these 16 classes - the confusion matrix for the validation set of these classes is shown below.

![Alt text](conf_mat.png?raw=true "Confusion Matrix")