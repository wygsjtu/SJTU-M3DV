# SJTU-M3DV
Final project for course EE369 in SJTU, implemented in Keras+Tensorflow.<br />
Model file 'dense.h5' is not in the file path, and can be downloaded it from [here](https://drive.google.com/open?id=1dGDvLPLX4haEq3ofeltP8ZARuycZI-PL).<br />
Four different neuron networks have been used to train the model, and the final model is trained via [data_train_densenet.py](https://github.com/wygsjtu/SJTU-M3DV/blob/master/data_train_densenet.py).<br />
## Requirements
- Python 3 (Anaconda 3.5.1 specifically)
- Tensorflow==1.14.0
- Keras==2.2.0
## Code Structure
[conv](https://github.com/wygsjtu/SJTU-M3DV/blob/master/conv): a simple implementation of 3D convolutional neuron network<br />
[densenet](https://github.com/wygsjtu/SJTU-M3DV/blob/master/densenet): an implementation of 3D Densenet network, 2017 CVPR best paper<br />
[resnet3d](https://github.com/wygsjtu/SJTU-M3DV/blob/master/resnet3d): an implementation of 3D Resnet network<br />
[inception](https://github.com/wygsjtu/SJTU-M3DV/blob/master/inception): an implementation of 3D Inception network<br />
[data](https://github.com/wygsjtu/SJTU-M3DV/blob/master/data): raw train data is not allowed to share in this repository.<br />
[data_train_cnn.py](https://github.com/wygsjtu/SJTU-M3DV/blob/master/data_train_cnn.py)<br />
[data_train_densenet.py](https://github.com/wygsjtu/SJTU-M3DV/blob/master/data_train_densenet.py)<br />
[data_train_resnet.py](https://github.com/wygsjtu/SJTU-M3DV/blob/master/data_train_resnet.py)<br />
[data_train_inc.py](https://github.com/wygsjtu/SJTU-M3DV/blob/master/data_train_inc.py)<br />
[misc.py](https://github.com/wygsjtu/SJTU-M3DV/blob/master/misc.py): basic functions<br />
[test.py](https://github.com/wygsjtu/SJTU-M3DV/blob/master/test.py): load model<br />
