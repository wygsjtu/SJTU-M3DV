# SJTU-M3DV
Final project for course EE369 in SJTU, implemented in Keras+Tensorflow
## Requirements
- Python 3 (Anaconda 3.5.1 specifically)
- Tensorflow==1.14.0
- Keras==2.2.0
## Code Structure
[conv](https://github.com/wygsjtu/SJTU-M3DV/conv): a simple implementation of 3D convolutional neuron network
[densenet](https://github.com/wygsjtu/SJTU-M3DV/densenet): an implementation of 3D Densenet network, 2017 CVPR best paper
[resnet3d](https://github.com/wygsjtu/SJTU-M3DV/resnet3d): an implementation of 3D Resnet network
[inception](https://github.com/wygsjtu/SJTU-M3DV/inception): an implementation of 3D Inception network
[data](https://github.com/wygsjtu/SJTU-M3DV/data): raw train data is not allowed to share in this repository.
[data_train_cnn.py](https://github.com/wygsjtu/SJTU-M3DV/data_train_cnn.py)
[data_train_densenet.py](https://github.com/wygsjtu/SJTU-M3DV/data_train_densenet.py)
[data_train_resnet.py](https://github.com/wygsjtu/SJTU-M3DV/data_train_resnet.py)
[data_train_inc.py](https://github.com/wygsjtu/SJTU-M3DV/data_train_inc.py)
[misc.py](https://github.com/wygsjtu/SJTU-M3DV/misc.py): basic functions
[test.py](https://github.com/wygsjtu/SJTU-M3DV/test.py): load model
[dense.h5](https://github.com/wygsjtu/SJTU-M3DV/dense.h5): saved model
