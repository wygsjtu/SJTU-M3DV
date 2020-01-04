import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from misc import calc_rate

# initialize
test_dir = './data/test/'
test_list = pd.read_table("./data/sample.csv",sep = ",")['name']
test_num = 117
MASK_SIZE = 32
test_data = np.ones((test_num, MASK_SIZE, MASK_SIZE, MASK_SIZE, 1))

# dataloader
for i in range(test_num):
    candidate = test_dir+test_list[i]+'.npz'
    data = np.load(candidate)
    test_data[i, :, :, :, 0] = \
	             data['voxel'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
				               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
							   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]*\
				 data['seg'][int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    			               int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2), \
    						   int((100-MASK_SIZE)/2):int((100+MASK_SIZE)/2)]
test_data = test_data.reshape(test_data.shape[0], MASK_SIZE, MASK_SIZE, MASK_SIZE, 1)
test_name = np.array(test_list).reshape(test_num)

# load model
new_model = load_model('dense.h5')
load_predict = new_model.predict(test_data)[:, 1]
load_predict = calc_rate(load_predict)
load_predicted = np.array(load_predict).reshape(test_num)
load_test_dict = {'Id':test_name, 'Predicted':load_predicted}
load_result = pd.DataFrame(load_test_dict, index = [0 for _ in range(test_num)])
load_result.to_csv("submission.csv", index = False, sep = ',')