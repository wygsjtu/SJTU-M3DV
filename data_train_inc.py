import numpy as np
import pandas as pd
import os
import csv
import random
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import misc

from inception.inception_3d import inception_3d
misc.set_gpu_usage()


BATCH_SIZE = 64
NUM_CLASSES = 2
NUM_EPOCHS = 30
MASK_SIZE = 32
train_val_num = 465
test_num = 117

train_dir = './data/train_val/'
test_dir = './data/test/'
train_list = os.listdir(train_dir)
test_list = pd.read_table("./data/sample.csv",sep = ",")['name']
label_list = pd.read_table("./data/train_val.csv",sep = ",")['diagnosis']
train_num = int(0.9*label_list.shape[0])
val_num = label_list.shape[0]-train_num
test_num = 117
data = np.load(train_dir+train_list[0])
train_data = np.ones((train_num, MASK_SIZE, MASK_SIZE, MASK_SIZE, 1))
train_label = []
val_data = np.ones((val_num, MASK_SIZE, MASK_SIZE, MASK_SIZE, 1))
val_label = []
test_data = np.ones((test_num, MASK_SIZE, MASK_SIZE, MASK_SIZE, 1))

for i in range(0, train_num):
    candidate = train_dir+train_list[i]
    data = np.load(candidate)
    train_data[i, :, :, :, 0] = \
	             data['voxel'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
				               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
							   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]*\
				 data['seg'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    			               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    						   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]
    train_label = np.append(train_label, label_list[i])
train_data = train_data.reshape(train_data.shape[0], MASK_SIZE, MASK_SIZE, MASK_SIZE, 1)
train_data = train_data.astype('float32') / 255.
train_label = keras.utils.to_categorical(train_label, NUM_CLASSES)

for i in range(train_num, label_list.shape[0]):
    candidate = train_dir+train_list[i]
    data = np.load(candidate)
    val_data[i-train_num, :, :, :, 0] = \
	             data['voxel'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
				               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
							   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]*\
				 data['seg'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    			               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    						   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]
    val_label = np.append(val_label, label_list[i])
val_data = val_data.reshape(val_data.shape[0], MASK_SIZE, MASK_SIZE, MASK_SIZE, 1)
val_data = val_data.astype('float32') / 255.
val_label = keras.utils.to_categorical(val_label, NUM_CLASSES)

for i in range(0, test_num):
    candidate = test_dir+test_list[i]+'.npz'
    data = np.load(candidate)
    test_data[i, :, :, :, 0] = \
	             data['voxel'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
				               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
							   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]*\
				 data['seg'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    			               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    						   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]
input_shape = (MASK_SIZE, MASK_SIZE, MASK_SIZE, 1)



model = inception_3d(2, input_shape, drop_rate=0.2)




# define the object function, optimizer and metrics
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['binary_accuracy'])
			  
checkpointer = ModelCheckpoint(filepath = './res/%s/weights.{epoch:02d}.h5' % 'test', verbose = 1,
                              period = 1, save_weights_only = False)
best_keeper = ModelCheckpoint(filepath = 'tmp/%s/best.h5' % 'test', verbose=1, save_weights_only = False,
                              monitor = 'binary_accuracy', save_best_only = True, period = 1, mode = 'max')
early_stopping = EarlyStopping(monitor = 'binary_accuracy', min_delta = 0, mode = 'max', patience = 30, verbose = 1)

model.fit(train_data, train_label, epochs = NUM_EPOCHS, batch_size = BATCH_SIZE)

score_train = model.evaluate(train_data, train_label)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (score_train[0], score_train[1]*100))
score_test = model.evaluate(val_data, val_label)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (score_test[0], score_test[1]*100))

test_name = np.array(test_list).reshape(test_num)
predict = model.predict(test_data)[:, 1]
predicted = np.array(predict).reshape(test_num)
test_dict = {'Id':test_name, 'Predicted':predicted}

result = pd.DataFrame(test_dict, index = [0 for _ in range(test_num)])

result.to_csv("result.csv", index = False, sep = ',')