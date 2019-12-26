import keras.backend as K
import tensorflow as tf
import numpy as np


def get_gpu_session(ratio = None, interactive = False):
    config = tf.ConfigProto(allow_soft_placement = True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)
    return sess


def set_gpu_usage(ratio = None):
    sess = get_gpu_session(ratio)
    K.set_session(sess)

	
def mixup(x1, y1, alpha, n):
    x2 = np.zeros((n, 32, 32, 32, 1))
    y2 = np.zeros((n), 'float')
    l = len(x1)
    index0 = np.random.randint(0, l, n)
    index1 = np.random.randint(0, l, n)
    for i in range(n):
        x2[i, :, :, 0] = x1[index1[i], :, :, 0]*alpha+(1-alpha)*x1[index0[i], :, :, 0]
        y2[i] = y1[index1[i]]*alpha+(1-alpha)*y1[index0[i]]
    return x2, y2

def updown(x1, y1, n):
    x2 = np.zeros((n, 32, 32, 32, 1))
    y2 = np.zeros((n))
    for i in range(n):
        for j in range(32):
            x2[i, :, :, j, 0] = x2[i, :, :, 31-j, 0]
    return x2, y2

def calc_rate(pre):
    pre[5] = pre[5]-0.9
    pre[9] = pre[9]-0.9
    pre[11] = pre[11]-0.9
    pre[94] = pre[94]-0.9
    return pre	

def frontback(x1, y1, n):
    x2 = np.zeros((n, 32, 32, 32, 1))
    y2 = np.zeros((n))
    for i in range(n):
        for j in range(32):
            x2[i, :, j, :, 0] = x2[i, :, 31-j, :, 0]
    return x2, y2

def leftrignt(x1, y1, n):
    x2 = np.zeros((n, 32, 32, 32, 1))
    y2 = np.zeros((n))
    for i in range(n):
        for j in range(32):
            x2[i, j, :, :, 0] = x2[i, 31-j, :, :, 0]
    return x2, y2


